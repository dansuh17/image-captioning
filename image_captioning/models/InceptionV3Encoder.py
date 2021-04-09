"""
Encoder using the Inception V3 backbone
"""
import tensorflow as tf


class InceptionV3Encoder(tf.keras.Model):
    """Encoder using the Inception V3 backbone
    """

    def __init__(self, embedding_dim: int, train_backbone: bool = False):
        """
        Arguments:
            embedding_dim: The number of channels to embed the output into.
            train_backbone: Whether to make backbone weights trainable
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Using Inception V3 backbone
        self.backbone = tf.keras.applications.InceptionV3(include_top=False,
                                                          weights='imagenet')
        self.backbone.trainable = train_backbone

        # FC layer to output into embedding dims
        self.fc = tf.keras.layers.Dense(embedding_dim)

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
        preprocessed_input = \
            tf.keras.applications.inception_v3.preprocess_input(images)
        features = self.backbone(preprocessed_input)
        features = tf.reshape(features,
                              (features.shape[0], -1, features.shape[3]))
        return tf.nn.relu(self.fc(features))

    def get_config(self) -> dict:
        """
            Returns:
                config: A dict providing the Encoder config info.
        """
        return {'embedding_dim': self.embedding_dim}
