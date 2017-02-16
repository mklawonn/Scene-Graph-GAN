#Code based on https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW/blob/master/main.py


import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope

import os
import numpy as np
import scipy.misc
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar

from gan import GAN


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch_size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan")


FLAGS = flags.FLAGS

if __name__ == "__main__":
    #TODO Figure out the actual data directory
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedires(data_directory)
    
    model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    #Main training loop
    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            #TODO change this portion of the code to read in a random batch of 
            #pre-computed VGG features paired with assertions
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            #TODO Generate images here
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss {}".format(training_loss))

        #TODO Make sure to change the generation from images to text
        #Note this will be done in the gan.py file
        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)
