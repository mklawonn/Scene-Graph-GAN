import os
import sys
import tensorflow as tf
import numpy as np


class Discriminator(object):

    def __init__(self, vocab_size, embedding_matrix):
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix

    def attentionMechanism(self, cell_state):
        context_and_state = tf.concat([self.flattened_context, cell_state[0]], axis=1)
        e = tf.layers.dense(inputs = context_and_state, units=self.conv3_5.get_shape()[1]*self.conv3_5.get_shape()[2], name="attention_perceptron", reuse=tf.AUTO_REUSE)
        self.alpha = tf.nn.softmax(e, name="attention_softmax")
        z_hat = tf.reduce_sum(tf.multiply(self.partially_flattened_context, tf.expand_dims(self.alpha, 2)), axis=1)
        return z_hat

    def loopFn(self, time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
            next_cell_state = self.cell_init_state
        else:
            next_cell_state = cell_state

        indices = self.input_triples[:, time-1, :]
        print indices
        embedding = tf.matmul(indices, self.embedding_matrix)
        next_input = tf.concat([self.attentionMechanism(self.cell_init_state), embedding], axis=1)

        next_loop_state = None
        elements_finished = (time >= 3)
        finished = tf.reduce_all(elements_finished)
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        
    def build_discriminator(self, input_triples, images, is_training=True):
        bias_init = tf.constant_initializer(0.05)
        kernel_init = tf.keras.initializers.he_normal()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        self.input_triples = input_triples

        ################################################## 
        # Convolutional Architecture
        ################################################## 
        #Block 1
        conv1_1 = tf.layers.conv2d(inputs = images, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv1_2 = tf.layers.conv2d(inputs = conv1_1, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm1_1 = tf.layers.batch_normalization(inputs = conv1_2, axis=-1, training=is_training)

        #Downsample
        conv1_3 = tf.layers.conv2d(inputs = batchnorm1_1, filters=32, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)

        #Block 2
        conv2_1 = tf.layers.conv2d(inputs = conv1_3, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv2_2 = tf.layers.conv2d(inputs = conv2_1, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm2_1 = tf.layers.batch_normalization(inputs = conv2_2, axis=-1, training=is_training)
        #Block 2_2
        conv2_3 = tf.layers.conv2d(inputs = batchnorm2_1, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv2_4 = tf.layers.conv2d(inputs = conv2_3, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm2_2 = tf.layers.batch_normalization(inputs = conv2_4, axis=-1, training=is_training)

        #Downsample
        conv2_5 = tf.layers.conv2d(inputs = batchnorm2_2, filters=128, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)

        #Block 3
        conv3_1 = tf.layers.conv2d(inputs = conv2_5, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv3_2 = tf.layers.conv2d(inputs = conv3_1, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm3_1 = tf.layers.batch_normalization(inputs = conv3_2, axis=-1, training=is_training)
        #Block 3_2
        conv3_3 = tf.layers.conv2d(inputs = batchnorm3_1, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        batchnorm3_2 = tf.layers.batch_normalization(inputs = conv3_4, axis=-1, training=is_training)

        #Downsample
        self.conv3_5 = tf.layers.conv2d(inputs = batchnorm3_2, filters=512, kernel_size=[5,5], padding="same", strides=2, activation=tf.nn.elu, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)

        ################################################## 
        # Process input triples with LSTM
        ################################################## 
        self.flattened_context = tf.reshape(self.conv3_5, [-1, self.conv3_5.get_shape()[1]*self.conv3_5.get_shape()[2]*self.conv3_5.get_shape()[3]])
        self.partially_flattened_context = tf.reshape(self.conv3_5, [-1, self.conv3_5.get_shape()[1]*self.conv3_5.get_shape()[2], self.conv3_5.get_shape()[3]])
        #TODO Should this also incorporate the triple somehow?
        self.cell_init_state = tf.reduce_mean(self.conv3_5, axis=(1,2))
        self.cell_init_state = tf.contrib.rnn.LSTMStateTuple(self.cell_init_state, self.cell_init_state)
        
        #LSTM receives as input at each timestep z_hat concat with
        #the triple input at that timestep
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(512)

        outputs_ta, final_state, decoded_logits = tf.nn.raw_rnn(lstm_cell, self.loopFn, swap_memory=True)

        lstm_outputs = tf.transpose(outputs_ta.stack(), perm=[1,0,2])

        discriminator_logits = tf.layers.dense(inputs = lstm_outputs, units=1, name="decoder", reuse=tf.AUTO_REUSE)

        return discriminator_logits
