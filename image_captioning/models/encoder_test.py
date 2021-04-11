"""
Simple test code for the Encoder module
"""
import random
import unittest

import tensorflow as tf

from .encoder import Encoder


class EncoderTest(unittest.TestCase):
    """Test Encoder model
    """

    def test_inception_v3_encoder_output_shape(self):
        """Test Inception V3 Encoder's output shape after call().
        """
        batch_size = random.randint(1, 10)
        embedding_dim = random.randint(1, 64)

        encoder = Encoder(model_name='inception_v3',
                          embedding_dim=embedding_dim)

        input_image = tf.random.uniform(shape=(batch_size, 299, 299, 3),
                                        dtype=tf.float32)

        output = encoder(input_image)

        self.assertEqual(output.shape, (batch_size, 64, embedding_dim))

    def test_inception_v3_encoder_get_config(self):
        """Test Inception V3 Encoder's get_config() content.
        """
        embedding_dim = random.randint(1, 64)
        model_name = 'inception_v3'

        encoder = Encoder(model_name=model_name, embedding_dim=embedding_dim)
        config = encoder.get_config()

        self.assertDictEqual(config, {
            'model_name': model_name,
            'embedding_dim': embedding_dim
        })


if __name__ == '__main__':
    unittest.main()
