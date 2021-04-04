"""
Attention models.
"""
from typing import Tuple

import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    """
    Implementation of the attention model described in the paper:
    Bahdanau et al. - "Neural Machine Translation by Jointly Learning to Align
    and Translate".

    The paper describes this as an "alignment model" that given the hidden
    state of a certain time step i, emphasizes another feature at time step j
    that has the most relevance.

    The `hidden_state` input to this layer is called `annotations`,
    but we're using a more common term used for learning RNN-based models
    to describe this.
    """
    def __init__(self, attn_hidden_units: int):
        """
        Arguments:
            attn_hidden_units: Number of units for the hidden layer.
                Not to be confused with the number of units in the
                hidden layer passed to the attention layer.
        """
        super().__init__()
        self.attn_hidden_units = attn_hidden_units  # for config

        # Define the hidden layers for this attention model.
        self.dense1 = tf.keras.layers.Dense(units=attn_hidden_units)
        self.dense2 = tf.keras.layers.Dense(units=attn_hidden_units)

        self.value_weight = tf.keras.layers.Dense(units=1)

    # pylint: disable=arguments-differ
    def call(self, features: tf.Tensor,
             hidden_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Arguments:
            features: Feature vector that contains a sequence of embeddings
                 to attend over the time (sequence) axis.
                `shape == (batch, time, embedding_dim)`.
            hidden_state: Hidden state that will be used as
                additional information to be broadcasted over the time axis
                before it gets passed along to the `value_weight`.
                `shape == (batch, hidden_units)`.

        Returns:
            context_vector: Embedding vectors summarized over the time axis,
                after being attended over time axis.
                `shape == (batch, embedding_dim)`.
            attention_weights: Attention weights or "scores".
                `shape == (batch, time=64, attn_weight=1)`
        """
        # features.shape == (batch, time, embedding_dim)
        # hidden_state.shape == (batch, hidden_units)

        # hidden_with_time_axis.shape == (batch, hidden_time=1, hidden_units)
        hidden_with_time_axis = tf.expand_dims(hidden_state, axis=1)

        # `hidden_with_time_axis` is broadcasted over time axis (axis 1).
        # attention_hidden_layer.shape == (batch, time, attn_hidden_units)
        attention_hidden_layer = tf.nn.tanh(
            self.dense1(features) + self.dense2(hidden_with_time_axis))

        # Scores per time.
        # score.shape == (batch, time, score=1)
        score = self.value_weight(attention_hidden_layer)

        # Attention score over time axis (axis 1).
        # attention_weights.shape == (batch, time, attn_weight=1)
        attention_weights = tf.nn.softmax(score, axis=1)

        ### ATTEND step.
        # Multiply the attention weights for each time step to the
        # corresponding embeddings.
        # context_vector.shape == (batch, time, embedding_dim)
        context_vector = attention_weights * features

        # Sum the embeddings over the time axis.
        # This gives the "context" vector, which contains a summary
        # over time represented as a single embedding vector per item.
        # context_vector.shape == (batch, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self) -> dict:
        """
        Returns:
            config: A dict providing the attention hidden units info.
        """
        return {'attn_hidden_units': self.attn_hidden_units}
