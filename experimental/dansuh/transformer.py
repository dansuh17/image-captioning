"""
Transformer practice by @dansuh17.
"""
import time
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds


def download_dataset(dataset_name="ted_hrlr_translate/pt_to_en"):
    examples, _ = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return examples["train"], examples["validation"]


def load_tokenizer(model_name="ted_hrlr_translate_pt_en_converter"):
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=".",
        cache_subdir="",
        extract=True,
    )
    return tf.saved_model.load(model_name)


def print_dataset_examples(
    trans_dataset: tf.data.Dataset, tokenizer_from, tokenizer_to, num_samples=3
):
    # gives tf.EagerTensor
    for from_examples, to_examples in trans_dataset.batch(num_samples).take(1):
        from_encoded = tokenizer_from.tokenize(from_examples)
        for i, from_item in enumerate(zip(from_examples.numpy(), from_encoded)):
            sentence, tokenized = from_item
            print(f'from-{i}: {sentence.decode("utf-8")}')
            print(f"    => {tokenized}")

        print()

        # RaggedTensor
        to_encoded = tokenizer_to.tokenize(to_examples)
        for i, to_item in enumerate(zip(to_examples.numpy(), to_encoded)):
            sentence, tokenized = to_item
            print(f'to-{i}: {sentence.decode("utf-8")}')
            print(f"    => {tokenized}")


def tokenize_pairs_func(from_tokenizer, to_tokenizer):
    def tokenize_pairs(from_data, to_data):
        from_tokenized = from_tokenizer.tokenize(from_data)
        to_tokenized = to_tokenizer.tokenize(to_data)
        return from_tokenized.to_tensor(), to_tokenized.to_tensor()

    return tokenize_pairs


def make_batches(dataset, tokenizers, buffer_size=20000, batch_size=64):
    tokenizer_from, tokenizer_to = tokenizers
    return (
        dataset.cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .map(
            tokenize_pairs_func(tokenizer_from, tokenizer_to),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def get_angles(pos, i, d_model):
    """
    Calculate the radians for positional encoding.

    w (angular frequency) = 1 / (10000 ^ (2 * floor(i / 2) / d_model))
    period = 1 / f = 2 * pi / w
        = 2 * pi * (10000 ^ (2 * floor(i / 2) / d_model))

    Interpretation:
    "The wavelengths [(periods)] form a geometric progression
    from 2 * pi to 10000 * 2 * pi."

    i is the dimension. It ranges from 0 to d_model - 1.
    The frequency is larger for smaller i (moves fast),
    and smaller for larger i (moves slow).

    Args:
        pos: position
        i: dimension
        d_model: model dimension

    Returns:
        Angles by positions.
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    # TODO: get the meaning of this
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i + 1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


print('*** Verify Positional Encoding')
pos_encoding_example = positional_encoding(position=2048, d_model=512)
print(pos_encoding_example.shape)
print(pos_encoding_example)
print()


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # (batch_size, 1, 1, seq_len)
    # Extra dimensions added to add padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int):
    """
    Sets upper triangle of a matrix to 1 of a size x size matrix.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
        q: query. shape == (..., seq_len_q, depth)
        k: key. shape == (..., seq_len_k, depth)
        v: value. shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to
            (..., seq_len_q, seq_len_k).  Defaults to None.

    Returns:
       output, attention_weights
    """
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # sacle matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # Softmax is normalized on the last axis (sqe_len_k) so that the
    # scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (..., sqe_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # query weights
        self.wq = tf.keras.layers.Dense(d_model)
        # key weights
        self.wk = tf.keras.layers.Dense(d_model)
        # value weights
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose so that output shape is:
            (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # out: (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split each into multiple heads.
        # out: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # The attention layer broadcasts the operation onto multiple heads.
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


print('*** Verify MultiHeadAttention')
tmp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
out, attn = tmp_mha(y, k=y, q=y, mask=None)
print(f'out.shape: {out.shape}')
print(f'attn.shape: {attn.shape}')


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    """
    Multi-Head Attention => Point-wise Feed Forward.
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Normalization done on the last axis.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(q=x, k=x, v=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        ### Attention 1
        # (batch_size, target_seq_len, d_model)
        # TODO: make sense of look_ahead_mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ### Attention 2
        # (batch_size, target_seq_len, d_model)
        # TODO: make sense of padding_mask
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ### Feed-Forward
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        max_position_encoding,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_encoding = positional_encoding(
            position=max_position_encoding, d_model=self.d_model
        )

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # x.shape == (batch_size, input_seq_len)
        seq_len = tf.shape(x)[1]

        # adding embedding and positional encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # (batch_size, input_seq_len, d_model)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        target_vocab_size: int,
        max_position_encoding: int,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            position=max_position_encoding, d_model=d_model
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # (batch_size, target_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            pe_input,
            dropout_rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            pe_target,
            dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    lr = (1 / sqrt(d_model)) * min((1 / sqrt(step)), step * warmup_steps ^ (-1.5))
    """

    def __init__(self, d_model: int, warmup_steps=4000):
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# TODO: make sense of this
def loss_function(real, pred, loss_obj):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_obj(real, pred)

    # apply padding mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    # apply padding mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# TODO: make sense
def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == "__main__":
    # hyper-parameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    checkpoint_path = "./checkpoints/train"
    epochs = 20

    train_dataset, val_dataset = download_dataset()
    tokenizers = load_tokenizer()

    print_dataset_examples(train_dataset, tokenizers.pt, tokenizers.en)

    train_batches = make_batches(
        train_dataset, tokenizers=(tokenizers.pt, tokenizers.en)
    )
    val_batches = make_batches(val_dataset, tokenizers=(tokenizers.pt, tokenizers.en))

    learning_rate = CustomLearningRateSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # TODO: make sense of SCC
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size(),
        target_vocab_size=tokenizers.en.get_vocab_size(),
        pe_input=1000,
        pe_target=1000,
        dropout_rate=dropout_rate,
    )

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Restore the latest checkpoint if exists.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    # TODO: "Teacher Forcing"
    # TODO: self-attention

    # TODO: what is this???
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]  # input
        tar_real = tar[:, 1:]  # target
        print(f'tar_inp: {tar_inp.shape}')
        print(f'tar_real: {tar_real.shape}')

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        print(f'enc_padding_mask: {enc_padding_mask.shape}')
        print(f'dec_padding_mask: {dec_padding_mask.shape}')
        print(f'combined_mask: {combined_mask.shape}')

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions, loss_obj)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
