"""
LSTM-based Decoder network.
"""
# Embedding vector of InceptionV3 = (batch, 8, 8, 2048) -> (batch, 64, 2048)
# Output of encoder: (batch, 64, embedding_dim) (after going through a Dense layer)
class Decoder(tf.keras.Model):
    """
    """
    def __init__(self, hidden_units: int, vocab_size: int, embedding_dim: int):
        """
        """
        super().__init__()
        # TODO
        self.hidden_units = hidden_units

        self.attention = BahdanauAttention(hidden_units)

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)

        self.lstm = tf.keras.layers.LSTM(
            units=self.hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_units)
        self.dense2 = tf.keras.layers.Dense(units=vocab_size)

    def call(self, prev_word, features, hidden_state):
        """
        """
        # prev_word = decoder input = previous word's index (teacher forcing)
        # prev_word.shape == (batch, 1)
        # features.shape == (batch, time=64, embedding_dim)
        # hidden_state.shape == (batch, hidden_units)

        # apply attention over L representation vectors
        # output after attention:
        #   context_vecctor: (batch, hidden_units)
        #   attention_weights: (batch, 64, 1)
        context_vector, attention_weights = self.attention(
            features, hidden_state)

        # (batch, 1, embedding_dim)
        x = self.embedding(prev_word)

        # Expand dims of context vector to make shape = (batch, 1, hidden_units).
        # After concat: (batch, 1, hidden_units + embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Output x.shape == (batch, 1, hidden_units)
        # final_memory_state.shape == (batch, hidden_units)
        #    This will be the next `hidden_state`.
        # final_carry_state.shape == (batch, hidden_units)
        x, final_memory_state, final_carry_state = self.lstm(x)

        # x.shape == (batch, 1, hidden_units)
        x = self.dense1(x)

        # x.shape == (batch, hidden_units)
        x = tf.reshape(x, (-1, x.shape[2]))

        # x.shape == (batch, vocab_size)
        x = self.dense2(x)

        return x, final_memory_state, attention_weights


if __name__ == '__main__':
    # Testing embedding layer
    embedding = tf.keras.layers.Embedding(10, 5)

    input_arr = tf.random.uniform(shape=[9, 1], maxval=10, dtype=tf.int32)
    print(input_arr.shape)
    print(embedding(input_arr).shape)

    decoder = Decoder(hidden_units=16, vocab_size=32, embedding_dim=64)

    prev_word = tf.random.uniform(shape=[8, 1], maxval=32, dtype=tf.int32)
    features = tf.random.uniform(shape=[8, 64, 64])
    hidden = tf.random.uniform(shape=[8, 16])
    word_pred, state, attn = decoder(prev_word, features, hidden)
    print(f'word_pred.shape: {word_pred.shape}')
    print(f'state.shape: {state.shape}')
    print(f'attn.shape: {attn.shape}')