from .attention import BahdanauAttention

import unittest
import tensorflow as tf


# Test BahdanauAttention model.
class BahadanauAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        attention = BahdanauAttention(attn_hidden_units=7)

        batch_size = 3
        time_size = 7
        embedding_dim = 11
        hidden_units = 5

        # Create dummy inputs.
        features = tf.random.uniform(
            shape=[batch_size, time_size, embedding_dim], dtype=tf.float32)
        hidden_state = tf.random.uniform(shape=[batch_size, hidden_units],
                                         dtype=tf.float32)

        # Pass through the attention layer.
        context_vector, attn_weights = attention(features, hidden_state)

        # Test whether the output shapes are as expected.
        self.assertEqual(context_vector.shape, (batch_size, embedding_dim))
        self.assertEqual(attn_weights.shape, (batch_size, time_size, 1))

    def test_get_config(self):
        attn_hidden_units = 89
        attention = BahdanauAttention(attn_hidden_units)
        config = attention.get_config()

        self.assertDictEqual(config, {'attn_hidden_units': attn_hidden_units})


if __name__ == '__main__':
    unittest.main()
