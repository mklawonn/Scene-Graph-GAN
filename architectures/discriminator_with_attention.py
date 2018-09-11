import os
import tensorflow as tf
import numpy as np


class Discriminator(object):

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def build_discriminator(self, images, input_triples):
        return tf.ones([2,2])
