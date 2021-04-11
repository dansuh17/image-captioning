"""
Encoder backbone selector function
"""
import tensorflow as tf

from .InceptionV3Encoder import InceptionV3Encoder


def select_model(model_name: str = 'inception_v3',
                 embedding_dim: int = 64,
                 train_backbone: bool = False) -> tf.keras.Model:
    """
    Arguments:
        model_name: The ame of backbone network to use as the Encoder.
        embedding_dim: The number of channels to embed the output into.
        train_backbone: Whether to make backbone weights trainable

    Returns:
        model: The tf.keras.Model of the Encoder specified by `model_name`.

    Raises:
        NotImplementedError: If the specified `model_name` is not implemented.
    """
    model_name = model_name.lower()
    if model_name == 'inception_v3':
        return InceptionV3Encoder(embedding_dim=embedding_dim,
                                  train_backbone=train_backbone)
    raise NotImplementedError
