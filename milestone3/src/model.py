#!python3
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from absl import logging

class Bertsum(tf.keras.Model):
    """
    Baseline implementation of the BertSum network found here:
        https://github.com/nlpyang/BertSum

    This is the transformer version of Bertsum
    """
    # TODO subclass non-transformer BertSum models from this class

    def __init__(self):
        super().__init__()
        # Model code here

    def call(self, inputs, training=False, **kwargs):
        # Forward pass code here
        # If batch norm / dropout is used, training MUST be passed
        # as an arg to these calls
        pass

    @staticmethod
    def load_bert():
        """
        Loads a pretrained BERT model suitable for use with BertSum.

        See the following reference for pretrained BERT on Tensorflow Hub:
            https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1

        Return: hub.KerasLayer
            A pretrained BERT model
        """
        # Import pretrained BERT from Tensorflow Hub or model files here
        return None
