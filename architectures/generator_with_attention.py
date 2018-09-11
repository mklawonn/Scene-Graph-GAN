import sys

import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self):

    def build_generator(self, images, training):
        bias_init = tf.constant_initializer(0.05)
        kernel_init = tf.keras.initializers.he_normal()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        ################################################## 
        # Convolutional Architecture
        ################################################## 
        #Block 1
        conv1_1 = tf.layers.conv2d(inputs = images, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv1_2 = tf.layers.conv2d(inputs = conv1_1, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm1_1 = tf.layers.batch_normalization(inputs = conv1_2, axis=-1, training=training)

        #Downsample
        conv1_3 = tf.layers.conv2d(inputs = batchnorm1_1, filters=32, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)

        #Block 2
        conv2_1 = tf.layers.conv2d(inputs = conv1_3, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv2_2 = tf.layers.conv2d(inputs = conv2_1, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm2_1 = tf.layers.batch_normalization(inputs = conv2_2, axis=-1, training=training)
        #Block 2_2
        conv2_3 = tf.layers.conv2d(inputs = batchnorm2_1, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv2_4 = tf.layers.conv2d(inputs = conv2_3, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm2_2 = tf.layers.batch_normalization(inputs = conv2_4, axis=-1, training=training)

        #Downsample
        conv2_5 = tf.layers.conv2d(inputs = batchnorm2_2, filters=128, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)

        #Block 3
        conv3_1 = tf.layers.conv2d(inputs = conv2_5, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv3_2 = tf.layers.conv2d(inputs = conv3_1, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm3_1 = tf.layers.batch_normalization(inputs = conv3_2, axis=-1, training=training)
        #Block 3_2
        conv3_3 = tf.layers.conv2d(inputs = batchnorm3_1, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm3_2 = tf.layers.batch_normalization(inputs = conv3_4, axis=-1, training=training)

        #Downsample
        conv3_5 = tf.layers.conv2d(inputs = batchnorm3_2, filters=512, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        
        ################################################## 
        # LSTM with Attention
        ################################################## 
        flattened_context = tf.reshape(conv3_5, [-1, conv3_5.get_shape()[1]*conv3_5.get_shape()[2]*conv3_5.get_shape()[3]])
        partially_flattened_context = tf.reshape(conv3_5, [-1, conv_3_5.get_shape()[1]*conv3_5.get_shape()[2], conv3_5.get_shape()[3]])
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(512)

        #Initialize state
        state = tf.reduce_mean(conv3_5, axis=(1,2))
        outputs = []
        for i in range(0, 3):
            #Calculate attention
            e = tf.layers.dense(inputs=flattened_context, units=conv3_5.get_shape()[1]*conv3_5.get_shape()[2], name="Attention perceptron", reuse=True)
            alpha = tf.nn.softmax(e, name="attention_softmax")
            print alpha 
            sys.exit(1)
            z_hat = tf.reduce_sum(tf.multiply(partially_flattened_context, tf.expand_dims(alpha, 2)), axis=1)
            #Feed dynamic z_hat into lstm cell
            output, state = lstm_cell.__call__(z_hat, state)
            outputs.append(output)

        return outputs
