"""
Encoder part of the image captioning module
"""
import tensorflow as tf

from .model_selector import select_model


class Encoder(tf.keras.Model):
    """Encoder wrapper for encoders of different backbones
    """

    def __init__(self,
                 model_name: str = 'inception_v3',
                 embedding_dim: int = 64):
        """
        Arguments:
            model_name: The ame of backbone network to use as the Encoder.
            embedding_dim: The number of channels to embed the output into.
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        self.model = select_model(model_name, embedding_dim)

    # pylint: disable=arguments-differ
    def call(self, images: tf.Tensor) -> tf.Tensor:
        """
        Arguments:
            images: The input image tensor.
                `shape == (batch, H, W, 3)`

        Returns:
            output: The encoded image embedding tensor.
                `shape == (batch, 64, embedding_dim)`
        """
        return self.model(images)

    def get_config(self) -> dict:
        """
            Returns:
                config: A dict providing the Encoder config info.
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
