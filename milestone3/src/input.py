#!python3
"""
Easy import of data here:
https://www.tensorflow.org/datasets/catalog/cnn_dailymail
"""

# TODO which of these imports can be removed
import logging
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Input

def load_cnn_dataset(**kwargs):
    """
    Loads, preprocesses, and returns the CNN/Daily Mail dataset as a
    tf.data.Dataset object

    Keyword Arguments:
        Add any flags needed to control preprocessing/loading here.
        Provide documentation for added flags

    Return: (Dataset, Dataset, Dataset)
        The preprocessed train, validation, and test sets for CNN/Daily Mail
    """
    logging.info("Loading CNN/Daily Mail dataset")
    # Code goes here
    return None, None, None

def preprocess(dataset):
    # Preprocessing steps go here, should perform steps given in BERTSUM paper
    # Feel free to change name/args, just call this method from load_cnn_dataset
    # Please provide a way to seed any random number generators
    pass
