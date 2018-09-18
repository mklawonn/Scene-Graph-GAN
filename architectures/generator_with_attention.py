import os, sys
sys.path.append(os.getcwd())


import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def attentionMechanism(self, cell_state):
        context_and_state = tf.concat([self.flattened_context, cell_state[0]], axis=1)
        e = tf.layers.dense(inputs = context_and_state, units=self.downsampled.get_shape()[1]*self.downsampled.get_shape()[2], name="attention_perceptron", reuse=tf.AUTO_REUSE)
        self.alpha = tf.nn.softmax(e, name="attention_softmax")
        z_hat = tf.reduce_sum(tf.multiply(self.partially_flattened_context, tf.expand_dims(self.alpha, 2)), axis=1)
        return z_hat

    def build_generator(self, images, is_training=True):
        bias_init = tf.constant_initializer(0.05)
        kernel_init = tf.keras.initializers.he_normal()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        ################################################## 
        # Convolutional Architecture
        ################################################## 
        #Block 1
        conv1_1 = tf.layers.conv2d(inputs = images, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm1_1 = tf.contrib.layers.layer_norm(inputs = conv1_1, activation_fn=tf.nn.elu)
        conv1_2 = tf.layers.conv2d(inputs = layernorm1_1, filters=32, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm1_2 = tf.contrib.layers.layer_norm(inputs = conv1_2, activation_fn=tf.nn.elu)

        #Downsample
        conv1_3 = tf.layers.conv2d(inputs = layernorm1_2, filters=32, kernel_size=[5,5], padding="same", strides=2, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm1_3 = tf.contrib.layers.layer_norm(inputs = conv1_3, activation_fn=tf.nn.elu)

        #Block 2
        conv2_1 = tf.layers.conv2d(inputs = layernorm1_3, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm2_1 = tf.contrib.layers.layer_norm(inputs = conv2_1, activation_fn=tf.nn.elu)
        conv2_2 = tf.layers.conv2d(inputs = layernorm2_1, filters=64, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm2_2 = tf.contrib.layers.layer_norm(inputs = conv2_2, activation_fn=tf.nn.elu)
        #Block 2_2
        conv2_3 = tf.layers.conv2d(inputs = layernorm2_2, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm2_3 = tf.contrib.layers.layer_norm(inputs = conv2_3, activation_fn=tf.nn.elu)
        conv2_4 = tf.layers.conv2d(inputs = layernorm2_3, filters=128, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm2_4 = tf.contrib.layers.layer_norm(inputs = conv2_4, activation_fn=tf.nn.elu)

        #Downsample
        conv2_5 = tf.layers.conv2d(inputs = layernorm2_4, filters=128, kernel_size=[5,5], padding="same", strides=2, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm2_5 = tf.contrib.layers.layer_norm(inputs = conv2_5, activation_fn=tf.nn.elu)

        #Block 3
        conv3_1 = tf.layers.conv2d(inputs = layernorm2_5, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm3_1 = tf.contrib.layers.layer_norm(inputs = conv3_1, activation_fn=tf.nn.elu)
        conv3_2 = tf.layers.conv2d(inputs = layernorm3_1, filters=256, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm3_2 = tf.contrib.layers.layer_norm(inputs = conv3_2, activation_fn=tf.nn.elu)
        #Block 3_2
        conv3_3 = tf.layers.conv2d(inputs = layernorm3_2, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm3_3 = tf.contrib.layers.layer_norm(inputs = conv3_3, activation_fn=tf.nn.elu)
        conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters=512, kernel_size=[3,3], padding="same", strides=1, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm3_4 = tf.contrib.layers.layer_norm(inputs = conv3_4, activation_fn=tf.nn.elu)

        #Downsample
        conv3_5 = tf.layers.conv2d(inputs = layernorm3_2, filters=512, kernel_size=[5,5], padding="same", strides=2, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        layernorm3_5 = tf.contrib.layers.layer_norm(inputs = conv3_5, activation_fn=tf.nn.elu)

        self.downsampled = tf.layers.conv2d(inputs = layernorm3_5, filters=512, kernel_size=[5,5], padding="same", strides=2, activation=None, kernel_regularizer=regularizer, bias_initializer = bias_init, kernel_initializer = kernel_init)
        
        ################################################## 
        # LSTM with Attention
        ################################################## 
        #TODO Check that inputs are being input in the correct order?
        self.flattened_context = tf.reshape(self.downsampled, [-1, self.downsampled.get_shape()[1]*self.downsampled.get_shape()[2]*self.downsampled.get_shape()[3]])
        self.partially_flattened_context = tf.reshape(self.downsampled, [-1, self.downsampled.get_shape()[1]*self.downsampled.get_shape()[2], self.downsampled.get_shape()[3]])
        state = tf.reduce_mean(self.downsampled, axis=(1,2))
        state = tf.contrib.rnn.LSTMStateTuple(state, state)

        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(512)

        noise = tf.random_normal(tf.shape(self.attentionMechanism(state)), mean=0.0, stddev=1.0)

        decoded_logits = []

        for i in range(3):
            next_input = tf.concat([self.attentionMechanism(state), noise], axis=1)
            output, state = lstm_cell(next_input, state)
            decoded_logits.append(tf.layers.dense(inputs = output, units=self.vocab_size, name="decoder", reuse=tf.AUTO_REUSE))

        decoded_logits = tf.stack(decoded_logits, axis=1)
        return decoded_logits
