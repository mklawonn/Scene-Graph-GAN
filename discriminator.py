import os
import tensorflow as tf
import numpy as np


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
        tf.set_random_seed(42)


    def build_discriminator(self, context_input, sequence_input):

        max_len = self.sequence_length
        #context_input = tf.placeholder(tf.float32, self.context_shape)
        #sequence_input = tf.placeholder(tf.float32, self.sequence_input_shape)
        #Context input has shape

        lstm_cell_size = 512
        #Shape batch_size, num_image_feats, image_feat_dim
        V = context_input
        #Embed each sequence word

        initializer = tf.random_uniform_initializer(-0.08, 0.08)
        #w_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.word_embedding_size], -0.08, 0.08))
        w_embed = tf.get_variable("w_embed", [self.vocab_size, self.word_embedding_size], initializer=initializer)
        #w_embed = tf.reshape(w_embed, [1, self.vocab_size, self.word_embedding_size])
        #w_embed = tf.tile(w_embed, [self.batch_size, 1, 1])
        #Q = tf.nn.embedding_lookup(w_embed, sequence_input)
        Q_flat = tf.matmul(tf.reshape(sequence_input, [self.batch_size*self.sequence_length, self.vocab_size]), w_embed)
        Q = tf.reshape(Q_flat, [self.batch_size, self.sequence_length, self.word_embedding_size])
        #To stay consistent with the paper's notation
        V_t = V
        Q_t = Q
        Q = tf.transpose(Q, perm=[0,2,1])
        V = tf.transpose(V, perm=[0,2,1])

        #Embed the whole sequence
        with tf.variable_scope("sequence_attention"):
            triple_embedder = tf.contrib.rnn.LSTMCell(lstm_cell_size, use_peepholes=True)
            initializer = tf.random_uniform_initializer(-0.08, 0.08)

            #Output is of shape [seq_length, batch_size, cell.output_size]
            output, _ = tf.nn.dynamic_rnn(triple_embedder, Q_t, dtype=tf.float32, swap_memory=True)
            #Q_t_unstacked = tf.unstack(Q_t, self.sequence_length, 1)
            #output, _ = tf.contrib.rnn.static_rnn(triple_embedder, Q_t_unstacked, dtype=tf.float32)
            #Output is now of shape batch_size, seq_length, cell.output_size
            #output = tf.transpose(output, perm=[1,0,2])
            #print output

            #Affinity weights for equation_3, but applied to triple_embedder LSTM outputs instead
            W_b_lstm = tf.get_variable("w_b_lstm", [lstm_cell_size, self.image_feat_dim], initializer=initializer)
            #Again to stay consistent with paper notation, we switch output and output transpose
            output_t = output
            output = tf.transpose(output, perm=[0,2,1])
            #C_lstm is the affinity matrix of shape batch_size, sequence_length, num_image_feats
            C_lstm_flat = tf.matmul(tf.reshape(output_t, [self.batch_size*self.sequence_length, lstm_cell_size]), W_b_lstm)
            C_lstm = tf.reshape(C_lstm_flat, [self.batch_size, self.sequence_length, self.image_feat_dim])
            #C_lstm is shape [batch_size, maxlen, num_image_feats]
            C_lstm = tf.tanh(tf.matmul(C_lstm, V))
            C_lstm_t = tf.transpose(C_lstm, perm=[0,2,1])

            #Setting up equation 4
            k=256
            W_v_lstm = tf.get_variable("w_v_lstm", [self.image_feat_dim, k], initializer=initializer)

            W_q_lstm = tf.get_variable("w_q_lstm", [lstm_cell_size, k], initializer=initializer)

            w_lstm_hv = tf.get_variable("w_lstm_hv", [k,1], initializer=initializer)

            w_lstm_hq = tf.get_variable("w_lstm_hq", [k,1], initializer=initializer)

            #Equation 4
            #Should be [batch_size, k, l]
            Hv_ = tf.matmul(tf.reshape(V, [self.batch_size*self.num_image_feats, self.image_feat_dim]), W_v_lstm)
            Hv_ = tf.transpose(tf.reshape(Hv_, [self.batch_size, self.num_image_feats, k]), perm=[0,2,1])
            #Should be [batch_size, k, sequence_length]
            Hq_ = tf.matmul(tf.reshape(output_t, [self.batch_size*self.sequence_length, lstm_cell_size]), W_q_lstm)
            Hq_ = tf.transpose(tf.reshape(Hq_, [self.batch_size, self.sequence_length, k]), perm=[0,2,1])
            Hv_lstm = tf.tanh(tf.add(Hv_, tf.matmul(Hq_, C_lstm)))
            Hq_lstm = tf.tanh(tf.add(Hq_, tf.matmul(Hv_, C_lstm_t)))


            #av is a batch of vectors of length self.num_image_feats
            #aq is a batch of vectors of length self.sequence_length
            Hv_lstm = tf.transpose(Hv_lstm, perm=[0,2,1])
            Hv_lstm = tf.reshape(Hv_lstm, [self.batch_size*self.num_image_feats, k])
            av_lstm = tf.nn.softmax(tf.matmul(Hv_lstm, w_lstm_hv))
            av_lstm = tf.reshape(av_lstm, [self.batch_size, 1, self.num_image_feats])
            Hq_lstm = tf.transpose(Hq_lstm, perm=[0,2,1])
            Hq_lstm = tf.reshape(Hq_lstm, [self.batch_size*self.sequence_length, k])
            aq_lstm = tf.nn.softmax(tf.matmul(Hq_lstm, w_lstm_hq))
            aq_lstm = tf.reshape(aq_lstm, [self.batch_size, 1, self.sequence_length])
            #Equation 5
            v_hats = tf.matmul(av_lstm, V, transpose_a=False, transpose_b=True)
            q_hats = tf.matmul(aq_lstm, output, transpose_a=False, transpose_b=True)

            v_hats = tf.reshape(v_hats, [self.batch_size, -1])
            q_hats = tf.reshape(q_hats, [self.batch_size, -1])

        with tf.variable_scope("word_attention"):
            initializer = tf.random_uniform_initializer(-0.08, 0.08)
            W_b = tf.get_variable("w_b", [self.word_embedding_size, self.image_feat_dim], initializer=initializer)
            #Again to stay consistent with paper notation, we switch output and output transpose
            Q_t = tf.transpose(Q, perm=[0,2,1])
            #Equation 3 in https://arxiv.org/pdf/1606.00061.pdf
            #C is the affinity matrix of shape batch_size, sequence_length, num_image_feats
            C_flat = tf.matmul(tf.reshape(Q_t, [self.batch_size*self.sequence_length, self.word_embedding_size]), W_b)
            C = tf.reshape(C_flat, [self.batch_size, self.sequence_length, self.image_feat_dim])
            #C_lstm is shape [batch_size, maxlen, num_image_feats]
            C = tf.tanh(tf.matmul(C, V))
            C_t = tf.transpose(C, perm=[0,2,1])

            #Setting up equation 4
            k=256
            W_v = tf.get_variable("w_v", [self.image_feat_dim, k], initializer=initializer)

            W_q = tf.get_variable("w_q", [self.word_embedding_size, k], initializer=initializer)

            w_hv = tf.get_variable("w_hv", [k,1], initializer=initializer)

            w_hq = tf.get_variable("w_hq", [k,1], initializer=initializer)

            #Equation 4 in https://arxiv.org/pdf/1606.00061.pdf
            #Should be [batch_size, k, l]
            Hv_ = tf.matmul(tf.reshape(V, [self.batch_size*self.num_image_feats, self.image_feat_dim]), W_v)
            Hv_ = tf.transpose(tf.reshape(Hv_, [self.batch_size, self.num_image_feats, k]), perm=[0,2,1])
            #Should be [batch_size, k, sequence_length]
            Hq_ = tf.matmul(tf.reshape(output_t, [self.batch_size*self.sequence_length, self.word_embedding_size]), W_q)
            Hq_ = tf.transpose(tf.reshape(Hq_, [self.batch_size, self.sequence_length, k]), perm=[0,2,1])
            Hv = tf.tanh(tf.add(Hv_, tf.matmul(Hq_, C)))
            Hq = tf.tanh(tf.add(Hq_, tf.matmul(Hv_, C_t)))

            #av is a batch of vectors of length self.num_image_feats
            #aq is a batch of vectors of length self.sequence_length
            Hv = tf.transpose(Hv, perm=[0,2,1])
            Hv = tf.reshape(Hv, [self.batch_size*self.num_image_feats, k])
            av = tf.nn.softmax(tf.matmul(Hv, w_hv))
            av = tf.reshape(av, [self.batch_size, 1, self.num_image_feats])
            Hq = tf.transpose(Hq, perm=[0,2,1])
            Hq = tf.reshape(Hq, [self.batch_size*self.sequence_length, k])
            aq = tf.nn.softmax(tf.matmul(Hq, w_hq))
            aq = tf.reshape(aq, [self.batch_size, 1, self.sequence_length])
            #Equation 5
            v_hatw = tf.matmul(av, V, transpose_a=False, transpose_b=True)
            q_hatw = tf.matmul(aq, output, transpose_a=False, transpose_b=True)

            v_hatw = tf.reshape(v_hatw, [self.batch_size, -1])
            q_hatw = tf.reshape(q_hatw, [self.batch_size, -1])

        #Setting up equation 7
        hw_dim = 256
        hs_dim = 128

        W_w = tf.get_variable("w_w", [self.image_feat_dim, hw_dim], initializer=initializer)
        b_w = tf.get_variable("b_w", [hw_dim], initializer=initializer)

        W_s = tf.get_variable("w_s", [hw_dim+int(v_hatw.get_shape()[1]), hs_dim], initializer=initializer)
        b_s = tf.get_variable("b_s", [hs_dim], initializer=initializer)

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
    vocab_size = 7621
    batch_size = 32
    d = Discriminator(vocab_size, batch_size = batch_size)
    #context_input = tf.get_variable("input_features", [batch_size, 196, 512])
    #sequence_input = tf.get_variable("sequence_input", [batch_size, 3, vocab_size])

    #context_input = np.zeros((batch_size, 196, 512), dtype=np.float32)
    #sequence_input = np.zeros((batch_size, 3, vocab_size), dtype=np.float32)
    context_input = np.load("ims.npy")
    sequence_input = np.load("triples.npy")
    logits = d.build_discriminator(context_input, sequence_input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print sess.run(output)
        print sess.run(logits)
