"""
Unit tests for Decoder model.
"""
import unittest

import tensorflow as tf

from .decoder import Decoder


class DecoderTest(unittest.TestCase):
    """Test cases for Decoder class.
    """

    def test_output_shape(self):
        """Test the output shapes of Decoder.
        """
        # Give different numbers (primes) to test better.
        batch_size = 3
        word_seq_len = 2
        feature_seq_len = 5
        vocab_size = 7
        hidden_units = 8
        attn_hidden_units = 11
        image_embedding_dim = 13
        word_embedding_dim = 17

        # Create dummy inputs
        # Note: Ignore some pylint options due to pylint false-positivies.
        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        prev_word = tf.random.uniform(shape=[batch_size, word_seq_len],
                                      maxval=vocab_size,
                                      dtype=tf.int32)

        features = tf.random.uniform(
            shape=[batch_size, feature_seq_len, image_embedding_dim],
            dtype=tf.float32)

        prev_hidden_state = tf.random.uniform(shape=[batch_size, hidden_units],
                                              dtype=tf.float32)

        decoder = Decoder(hidden_units, attn_hidden_units, vocab_size,
                          word_embedding_dim)

        # Pass the inputs through the decoder.
        seq_out, final_memory_state, attention_weights = decoder(
            prev_word, features, prev_hidden_state)

        # Test output sizes.
        self.assertEqual(seq_out.shape, (batch_size, word_seq_len, vocab_size))
        self.assertEqual(final_memory_state.shape, (batch_size, hidden_units))
        self.assertEqual(attention_weights.shape,
                         (batch_size, feature_seq_len, 1))

    def test_get_config(self):
        """Test the get_config method of Decoder.
        """
        vocab_size = 7
        hidden_units = 8
        attn_hidden_units = 11
        word_embedding_dim = 13

        decoder = Decoder(hidden_units, attn_hidden_units, vocab_size,
                          word_embedding_dim)

        # Test the equality of config.
        self.assertDictEqual(
            decoder.get_config(), {
                'hidden_units': hidden_units,
                'vocab_size': vocab_size,
                'word_embedding_dim': word_embedding_dim,
                'attn_hidden_units': attn_hidden_units,
            })


if __name__ == '__main__':
    unittest.main()
