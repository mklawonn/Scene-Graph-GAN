import os
import tensorflow as tf
import numpy as np

#from tensorflow.models.rnn import rnn, rnn_cell



class Discriminator(object):
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, vocab_size, dim_lstm_hidden=256, batch_size=64, 
                context_shape=[196, 512], 
                word_embedding_size=512, bias_init_vector=None):

        #Placeholders
        #self.context_input = tf.placeholder(tf.float32, context_shape)
        #self.sequence_input = tf.placeholder(tf.int32, sequence_input_shape)
        self.context_shape = [batch_size, context_shape[0], context_shape[1]]
        self.sequence_input_shape = [batch_size, 3, vocab_size]

        self.word_embedding_size = word_embedding_size
        
        self.vocab_size = vocab_size
        self.num_image_feats = context_shape[0]
        self.image_feat_dim = context_shape[1]
        self.sequence_length = self.sequence_input_shape[1]
        self.batch_size = batch_size


    def build_discriminator(self, context_input, sequence_input):

        max_len = self.sequence_length
        #context_input = tf.placeholder(tf.float32, self.context_shape)
        #sequence_input = tf.placeholder(tf.float32, self.sequence_input_shape)
        #Context input has shape

        lstm_cell_size = 512
        V = context_input
        #Embed each sequence word

        initializer = tf.random_uniform_initializer(-0.08, 0.08)
        #w_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.word_embedding_size], -0.08, 0.08))
        w_embed = tf.get_variable("w_embed", [self.vocab_size, self.word_embedding_size], initializer=initializer)
        w_embed = tf.reshape(w_embed, [1, self.vocab_size, self.word_embedding_size])
        w_embed = tf.tile(w_embed, [self.batch_size, 1, 1])
        #Q = tf.nn.embedding_lookup(w_embed, sequence_input)
        Q = tf.matmul(sequence_input, w_embed)
        #To stay consistent with the paper's notation
        Q = tf.transpose(Q, perm=[0,2,1])
        V = tf.transpose(V, perm=[0,2,1])

        #Embed the whole sequence
        with tf.variable_scope("sequence_attention"):
            triple_embedder = tf.contrib.rnn.LSTMCell(lstm_cell_size, use_peepholes=True)
            initializer = tf.random_uniform_initializer(-0.08, 0.08)
            #triple_embedder = tf.contrib.rnn.MultiRNNCell([triple_embedder] * 2)
            output, _ = tf.nn.dynamic_rnn(triple_embedder, Q, dtype=tf.float32, swap_memory=True)
            #Output is of shape [batch_size, max_time, cell.output_size]
            #output = tf.transpose(output, [1, 0, 2])
            #Affinity weights for equation_3, but applied to triple_embedder LSTM outputs instead
            #W_b_lstm = tf.Variable(tf.random_uniform([lstm_cell_size, self.image_feat_dim], -0.08, 0.08))
            W_b_lstm = tf.get_variable("w_b_lstm", [lstm_cell_size, self.image_feat_dim], initializer=initializer)
            W_b_lstm = tf.reshape(W_b_lstm, [1, lstm_cell_size, self.image_feat_dim])
            W_b_lstm = tf.tile(W_b_lstm, [self.batch_size, 1, 1])
            #Again to stay consistent with paper notation, we switch output and output transpose
            output_t = output
            output = tf.transpose(output, perm=[0,2,1])
            #C_lstm is the affinity matrix of shape batch_size, sequence_length, num_image_feats
            C_lstm = tf.tanh(tf.matmul(tf.matmul(output_t, W_b_lstm), V))
            C_lstm_t = tf.transpose(C_lstm, perm=[0,2,1])
            k=256
            #Setting up equation 4
            #W_v_lstm = tf.Variable(tf.random_uniform([k, self.image_feat_dim], -0.08, -0.08))
            W_v_lstm = tf.get_variable("w_v_lstm", [k, self.image_feat_dim], initializer=initializer)
            W_v_lstm_exp = tf.reshape(W_v_lstm, [1, k, self.image_feat_dim])
            W_v_lstm_tile = tf.tile(W_v_lstm_exp, [self.batch_size, 1, 1])
            #W_q_lstm = tf.Variable(tf.random_uniform([k, lstm_cell_size]))
            W_q_lstm = tf.get_variable("w_q_lstm", [k, lstm_cell_size], initializer=initializer)
            W_q_lstm_exp = tf.reshape(W_q_lstm, [1, k, lstm_cell_size])
            W_q_lstm_tile = tf.tile(W_q_lstm_exp, [self.batch_size, 1, 1])
            #w_lstm_hv = tf.Variable(tf.random_uniform([k], -0.08, 0.08))
            w_lstm_hv = tf.get_variable("w_lstm_hv", [k], initializer=initializer)
            w_lstm_hv = tf.reshape(w_lstm_hv, [1, 1, k])
            w_lstm_hv = tf.tile(w_lstm_hv, [self.batch_size, 1, 1])
            #w_lstm_hq = tf.Variable(tf.random_uniform([k], -0.08, 0.08))
            w_lstm_hq = tf.get_variable("w_lstm_hq", [k], initializer=initializer)
            w_lstm_hq = tf.reshape(w_lstm_hq, [1, 1, k])
            w_lstm_hq = tf.tile(w_lstm_hq, [self.batch_size, 1, 1])
            #Equation 4
            Hv_lstm = tf.tanh(tf.add(tf.matmul(W_v_lstm_tile, V), tf.matmul(tf.matmul(W_q_lstm_tile, output), C_lstm)))
            Hq_lstm = tf.tanh(tf.add(tf.matmul(W_q_lstm_tile, output), tf.matmul(tf.matmul(W_v_lstm_tile, V), C_lstm_t)))

            av_lstm = tf.nn.softmax(tf.matmul(w_lstm_hv, Hv_lstm, transpose_a=False))
            aq_lstm = tf.nn.softmax(tf.matmul(w_lstm_hq, Hq_lstm, transpose_a=False))
            #Equation 5
            v_hats = tf.matmul(av_lstm, V, transpose_a=False, transpose_b=True)
            q_hats = tf.matmul(aq_lstm, output, transpose_a=False, transpose_b=True)

            v_hats = tf.reshape(v_hats, [self.batch_size, -1])
            q_hats = tf.reshape(q_hats, [self.batch_size, -1])




        with tf.variable_scope("word_attention"):
            initializer = tf.random_uniform_initializer(-0.08, 0.08)
            #The affinity weights for equation 3
            #W_b = tf.Variable(tf.random_uniform([self.word_embedding_size, self.image_feat_dim], -0.08, 0.08))
            W_b = tf.get_variable("w_b", [self.word_embedding_size, self.image_feat_dim], initializer=initializer)
            W_b = tf.reshape(W_b, [1, self.word_embedding_size, self.image_feat_dim])
            W_b = tf.tile(W_b, [self.batch_size, 1, 1])

            
            Q_t = tf.transpose(Q, perm=[0,2,1])
            #Equation 3 in https://arxiv.org/pdf/1606.00061.pdf
            #C is the affinity matrix of shape batch_size, sequence_length, num_image_feats
            C = tf.tanh(tf.matmul(tf.matmul(Q_t, W_b), V))
            C_t = tf.transpose(C, perm=[0,2,1])

            #Setting up equation 4
            #k represents the 
            k = 256
            #W_v = tf.Variable(tf.random_uniform([k, self.image_feat_dim], -0.08, -0.08))
            W_v = tf.get_variable("w_v", [k, self.image_feat_dim], initializer=initializer)
            W_v_exp = tf.reshape(W_v, [1, k, self.image_feat_dim])
            W_v_tile = tf.tile(W_v_exp, [self.batch_size, 1, 1])
            #W_q = tf.Variable(tf.random_uniform([k, self.word_embedding_size], -0.08, -0.08))
            W_q = tf.get_variable("w_q", [k, self.word_embedding_size], initializer=initializer)
            W_q_exp = tf.reshape(W_q, [1, k, self.word_embedding_size])
            W_q_tile = tf.tile(W_q_exp, [self.batch_size, 1, 1])
            #w_hv = tf.Variable(tf.random_uniform([k], -0.08, 0.08))
            w_hv = tf.get_variable("w_hv", [k], initializer=initializer)
            w_hv = tf.reshape(w_hv, [1, 1, k])
            w_hv = tf.tile(w_hv, [self.batch_size, 1, 1])
            #w_hq = tf.Variable(tf.random_uniform([k], -0.08, 0.08))
            w_hq = tf.get_variable("w_hq", [k], initializer=initializer)
            w_hq = tf.reshape(w_hq, [1, 1, k])
            w_hq = tf.tile(w_hq, [self.batch_size, 1, 1])

            #Equation 4 in https://arxiv.org/pdf/1606.00061.pdf
            Hv = tf.tanh(tf.add(tf.matmul(W_v_tile, V), tf.matmul(tf.matmul(W_q_tile, Q), C)))
            Hq = tf.tanh(tf.add(tf.matmul(W_q_tile, Q), tf.matmul(tf.matmul(W_v_tile, V), C_t)))
            
            #av is a batch of vectors of length self.num_image_feats
            #aq is a batch of vectors of length self.sequence_length
            av = tf.nn.softmax(tf.matmul(w_hv, Hv, transpose_a=False))
            aq = tf.nn.softmax(tf.matmul(w_hq, Hq, transpose_a=False))

            #Equation 5
            v_hatw = tf.matmul(av, V, transpose_a=False, transpose_b=True)
            q_hatw = tf.matmul(aq, Q, transpose_a=False, transpose_b=True)

            v_hatw = tf.reshape(v_hatw, [self.batch_size, -1])
            q_hatw = tf.reshape(q_hatw, [self.batch_size, -1])

        #Setting up equation 7
        hw_dim = 256
        hs_dim = 128
        #W_w = tf.Variable(tf.random_uniform([self.image_feat_dim, hw_dim], -0.08, 0.08))
        #b_w = tf.Variable(tf.random_uniform([hw_dim], -0.08, 0.08))
        W_w = tf.get_variable("w_w", [self.image_feat_dim, hw_dim], initializer=initializer)
        b_w = tf.get_variable("b_w", [hw_dim], initializer=initializer)
        #W_s = tf.Variable(tf.random_uniform([hw_dim+int(v_hatw.get_shape()[1]), hs_dim], -0.08, 0.08))
        #b_s = tf.Variable(tf.random_uniform([hs_dim], -0.08, 0.08))
        W_s = tf.get_variable("w_s", [hw_dim+int(v_hatw.get_shape()[1]), hs_dim], initializer=initializer)
        b_s = tf.get_variable("b_s", [hs_dim], initializer=initializer)
        #W_h = tf.Variable(tf.random_uniform([hs_dim, 1], -0.08, 0.08))
        #b_h = tf.Variable(tf.random_uniform([1], -0.08, 0.08))
        W_h = tf.get_variable("w_h", [hs_dim, 1], initializer=initializer)
        b_h = tf.get_variable("b_h", [1], initializer=initializer)

        #Equation 7
        hw = tf.tanh(tf.add(tf.matmul(tf.add(q_hatw, v_hatw), W_w), b_w))
        #We don't convolve on unigrams, bigrams, and trigrams because the input sequence is so short
        #hp = tf.tanh(tf.matmul(W_p, tf.concat(tf.add(q_hatp, v_hatp), hw)))
        added = tf.add(q_hats, v_hats)
        concat = tf.concat([added, hw], 1)
        hs = tf.tanh(tf.add(tf.matmul(concat, W_s), b_s))
        logits = tf.add(tf.matmul(hs, W_h), b_h)
        #out = tf.sigmoid(tf.matmul(W_h, hs))
        out = tf.sigmoid(logits)
        #Return the logits for the wasserstein gan
        return logits


if __name__ == "__main__":
    vocab_size = 30000
    batch_size = 32
    d = Discriminator(vocab_size, batch_size = batch_size)
    contex_input = tf.get_variable("input_features", [batch_size, 196, 512])
    sequence_input = tf.get_variable("sequence_input", [batch_size, 3, vocab_size])
    logits = d.build_discriminator(contex_input, sequence_input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(logits)
