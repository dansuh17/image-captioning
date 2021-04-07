"""
Decoder model.
"""
from typing import Tuple

import tensorflow as tf

from .attention import BahdanauAttention


class Decoder(tf.keras.Model):
    """Decoder model.
    """

    def __init__(self, hidden_units: int, attn_hidden_units: int,
                 vocab_size: int, word_embedding_dim: int):
        """
        Arguments:
            hidden_units: Number of hidden units for LSTM.
            attn_hidden_units: Number of hidden units for attention module.
            vocab_size: Number of possible vocabularies in the language.
            word_embedding_dim: Dimension of word embedding vector.
        """
        super().__init__(self)

        self.hidden_units = hidden_units
        self.attn_hidden_units = attn_hidden_units
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim

        self.attention = BahdanauAttention(attn_hidden_units)

        # Convert word indices to embedding vectors.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=word_embedding_dim)

        self.lstm = tf.keras.layers.LSTM(units=self.hidden_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.dense1 = tf.keras.layers.Dense(units=hidden_units)

        # Final layer to convert into "scores" per each vocabulary.
        self.dense2 = tf.keras.layers.Dense(units=vocab_size)

    # pylint: disable=arguments-differ
    def call(
            self, prev_word: tf.Tensor, features: tf.Tensor,
            prev_hidden_state: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Arguments:
            prev_word: Sequence of indices of previous words,
                used as ingredients to predict the next `word_seq_len` words.
                `shape == (batch, word_seq_len)`.
            features: Sequence of feature embeddings for the corresponding
                image.
                `shape == (batch, feature_seq_len, image_embedding_dim)`.
            prev_hidden_state: Previous hidden state (`final_memory_state`).
                `shape == (batch, hidden_units)`.

        Returns:
            seq_out: Output scores for each word vocabulary.
                `shape == (batch, word_seq_len, vocab_size)`
            final_memory_state: Final memory state for LSTM module.
                `shape == (batch, hidden_units)`
            attention_weights: Attention weights that were used to create
                the context vector.
                `shape == (batch, feature_seq_len, attn_weight=1)`
        """
        # context_vector.shape == (batch, image_embedding_dim)
        # attention_weights.shape == (batch, feature_seq_len, attn_weight=1)
        context_vector, attention_weights = self.attention(
            features, prev_hidden_state)

        # Create word embeddings
        # x.shape == (batch, word_seq_len, word_embedding_dim)
        word_embeddings = self.embedding(prev_word)

        # Add a time axis to the context vector and repeat the context
        # vector over the length of time so that it can be concatenated
        # with `word_embeddings`.
        # context_vector.shape == (batch, word_seq_len, image_embedding_dim)
        context_vector_target_shape = (
            word_embeddings.shape[0],  # batch
            word_embeddings.shape[1],  # word_seq_len
            context_vector.shape[1])  # image_embedding_dim
        context_vector = tf.broadcast_to(tf.expand_dims(context_vector, axis=1),
                                         context_vector_target_shape)

        # Attach context vectors to extend each of word embeddings.
        # word_embeddings.shape == (batch, word_seq_len,
        # image_embedding_dim + word_embedding_dim)
        #
        # Note: Ignore some pylint options due to pylint false-positivies.
        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        word_embeddings = tf.concat(values=[context_vector, word_embeddings],
                                    axis=-1)

        # seq_out.shape == (batch, word_seq_len, hidden_units)
        # final_memory_state.shape == (batch, hidden_units)
        #   This will be the next `hidden state`.
        # final_carry_state.shape == (batch, hidden_units)
        seq_out, final_memory_state, _ = self.lstm(word_embeddings)

        # seq_out.shape == (batch, word_seq_len, hidden_units)
        seq_out = self.dense1(seq_out)

        # seq_out.shape == (batch, word_seq_len, vocab_size)
        seq_out = self.dense2(seq_out)

        return seq_out, final_memory_state, attention_weights

    def get_config(self) -> dict:
        """
        Returns:
            config: Configuration of Decoder instance.
        """
        return {
            'hidden_units': self.hidden_units,
            'vocab_size': self.vocab_size,
            'word_embedding_dim': self.word_embedding_dim,
            'attn_hidden_units': self.attn_hidden_units,
        }
