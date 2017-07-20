import os
import tensorflow as tf
import numpy as np


class Discriminator(object):

    def __init__(self, vocab_size, dim_embed=300, dim_context=512, dim_hidden=512, n_lstm_steps=3, batch_size=64, context_shape=[196,512], bias_init_vector=None):
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.context_shape = context_shape
        self.n_lstm_steps = n_lstm_steps
        #self.batch_size = batch_size
        self.maxlen = n_lstm_steps
        self.flag_shape = 512

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        constant_initializer = tf.constant_initializer(0.05)
        alpha_initializer = tf.constant_initializer(1)
        beta_initializer = tf.constant_initializer(0)

        #The weights for encoding the context to feed it into the attention MLP
        #Needs to be encoded in order to combine it with the previous hidden state
        self.context_encode_W = tf.get_variable("context_encode_W", [self.context_shape[0]*self.context_shape[1], self.dim_context*2], initializer=xavier_initializer)
        self.context_encode_b = tf.get_variable("context_encode_b", [self.dim_context*2], initializer=constant_initializer)

        #self.flag_encode_W = tf.get_variable("flag_encode_W", [self.flag_shape, self.context_shape[1]], initializer=xavier_initializer)
        #self.flag_encode_b = tf.get_variable("flag_encode_b", [self.context_shape[1]], initializer=constant_initializer)

        self.Wemb = tf.get_variable("Wemb", [self.vocab_size, self.dim_embed], initializer=xavier_initializer)

        self.att_W = tf.get_variable("att_W", [(self.dim_context*2)+self.dim_hidden, self.context_shape[0]], initializer=xavier_initializer)
        self.att_b = tf.get_variable("att_b", [self.context_shape[0]], initializer=constant_initializer)

        self.lstm_W = tf.get_variable("lstm_W", [self.dim_context+self.dim_hidden+self.dim_embed+self.flag_shape, self.dim_hidden*4], initializer=xavier_initializer)
        self.lstm_b = tf.get_variable("lstm_b", [self.dim_hidden*4], initializer=constant_initializer)

        #self.init_hidden_W = tf.get_variable("init_hidden_W", [self.dim_context, self.dim_hidden], initializer=xavier_initializer)
        #self.init_hidden_b = tf.get_variable("init_hidden_b", [self.dim_hidden], initializer=constant_initializer)

        #self.init_memory_W = tf.get_variable("init_memory_W", [self.dim_context, self.dim_hidden], initializer=xavier_initializer)
        #self.init_memory_b = tf.get_variable("init_memory_b", [self.dim_hidden], initializer=constant_initializer)

        self.init_hidden_W = tf.get_variable("init_hidden_W", [self.dim_context + self.vocab_size, self.dim_hidden], initializer=xavier_initializer)
        self.init_hidden_b = tf.get_variable("init_hidden_b", [self.dim_hidden], initializer=constant_initializer)

        self.init_memory_W = tf.get_variable("init_memory_W", [self.dim_context + self.vocab_size, self.dim_hidden], initializer=xavier_initializer)
        self.init_memory_b = tf.get_variable("init_memory_b", [self.dim_hidden], initializer=constant_initializer)

        self.decode_lstm_W = tf.get_variable("decode_lstm_W", [self.dim_hidden, 1], initializer=xavier_initializer)
        self.decode_lstm_b = tf.get_variable("decode_lstm_b", [1], initializer=constant_initializer)

        #Uncomment the next four lines of code for a deep output layer
        #self.decode_lstm_W = tf.get_variable("decode_lstm_W", [self.dim_hidden, self.dim_embed], initializer=xavier_initializer)
        #self.decode_lstm_b = tf.get_variable("decode_lstm_b", [self.dim_embed], initializer=constant_initializer)

        #self.decode_word_W = tf.get_variable("decode_word_W", [self.dim_embed, 1], initializer=xavier_initializer)
        #self.decode_word_b = tf.get_variable("decode_word_b", [1], initializer=constant_initializer)

        #For layer normalization:
        self.init_hidden_alpha = tf.get_variable("init_hidden/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.init_hidden_beta = tf.get_variable("init_hidden/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.init_memory_alpha = tf.get_variable("init_memory/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.init_memory_beta = tf.get_variable("init_memory/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.encoded_context_alpha = tf.get_variable("encoded_context/LN/alpha", shape=[self.context_shape[1]*2], initializer = alpha_initializer)
        self.encoded_context_beta = tf.get_variable("encoded_context/LN/beta", shape=[self.context_shape[1]*2], initializer = beta_initializer)

        self.i_alpha = tf.get_variable("i/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.i_beta = tf.get_variable("i/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.f_alpha = tf.get_variable("f/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.f_beta = tf.get_variable("f/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.o_alpha = tf.get_variable("o/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.o_beta = tf.get_variable("o/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.g_alpha = tf.get_variable("g/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.g_beta = tf.get_variable("g/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)


        
    def layer_normalization(self, inputs, scale, shift, epsilon = 1e-5):
        mean, var = tf.nn.moments(inputs, [1], keep_dims = True)

        normalized = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
        return normalized


    def get_initial_lstm(self, mean_context, first_word):
        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This is done via the mean_context ops
        context_and_word = tf.concat([mean_context, first_word], axis=1)
        initial_hidden = tf.add(tf.matmul(context_and_word, self.init_hidden_W), self.init_hidden_b)
        initial_hidden = self.layer_normalization(initial_hidden, self.init_hidden_alpha, self.init_hidden_beta)
        initial_hidden = tf.nn.tanh(initial_hidden)
        initial_memory = tf.add(tf.matmul(context_and_word, self.init_memory_W), self.init_memory_b)
        initial_memory = self.layer_normalization(initial_memory, self.init_memory_alpha, self.init_memory_beta)
        initial_memory = tf.nn.tanh(initial_memory)

        return initial_hidden, initial_memory

    def build_discriminator(self, context, input_triples, batch_size, attributes_flag):
        #From the paper: The initial memory state and hidden state of the LSTM are predicted
        #by an average of the annotation vectors fed through two separate MLPs
        first_word = input_triples[:, 0, :]
        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1), first_word)#(batch_size, dim_hidden)

        flag = tf.reshape(attributes_flag, [batch_size, 1])
        flag = tf.tile(flag, [1, self.flag_shape])
        #embedded_flag = tf.add(tf.matmul(flag, self.flag_encode_W), self.flag_encode_b)

        flattened_context = tf.reshape(context, [-1, self.context_shape[0]*self.context_shape[1]])
        encoded_context = tf.add(tf.matmul(flattened_context, self.context_encode_W), self.context_encode_b)
        encoded_context = self.layer_normalization(encoded_context, self.encoded_context_alpha, self.encoded_context_beta)
        
        logits_list = []

        for ind in range(self.n_lstm_steps):
            current_words = input_triples[:, ind, :]
            current_words = tf.reshape(current_words, [-1, self.vocab_size])
            word_emb = tf.matmul(current_words, self.Wemb)
            #Calculate \hat{z}
            #Equation (4)
            #Concatenate flattened context with previous hidden state and the noise vector, and feed through attention mlp
            #encoded_context = tf.nn.relu(encoded_context)
            context_and_hidden_state = tf.concat([encoded_context, h], 1)
            e = tf.add(tf.matmul(context_and_hidden_state, self.att_W), self.att_b)
            #Equation (5)
            alpha = tf.nn.softmax(e)
            #Equation (6) and (13): Soft attention model
            z_hat = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) #Output is [batch size, D]
            #Equation (1)
            #Concatenate \hat{z}_t , h_{t-1}, and the embedding of the previous word 
            lstm_input = tf.concat([z_hat, h, word_emb, flag], 1)
            #Perform affine transformation of concatenated vector
            affine = tf.add(tf.matmul(lstm_input, self.lstm_W), self.lstm_b)
            i, f, o, g = tf.split(affine, 4, 1)
            #i_t = sigmoid(affine)
            i = self.layer_normalization(i, self.i_alpha, self.i_beta)
            i = tf.nn.sigmoid(i)
            #f_t = sigmoid(affine)
            f = self.layer_normalization(f, self.f_alpha, self.f_beta)
            f = tf.nn.sigmoid(f)
            #o_t = sigmoid(affine)
            o = self.layer_normalization(o, self.o_alpha, self.o_beta)
            o = tf.nn.sigmoid(o)
            #g_t = tanh(affine)
            g = self.layer_normalization(g, self.g_alpha, self.g_beta)
            g = tf.nn.tanh(g)

            #Equation (2)
            #c_t = f_t * c_{t-1} + i_t * g_t
            c = f * c + i * g
            #h_t = o_t * tanh(c_t)
            h = o * tf.nn.tanh(c)

            logits = tf.add(tf.matmul(h, self.decode_lstm_W), self.decode_lstm_b)

            #logit_words = tf.add(tf.matmul(logits, self.decode_word_W), self.decode_word_b)

            logits_list.append(logits)
            

        #all_logits = tf.stack(logits_list)
        #We want it in shape [batch_size, seq_len, 1]
        all_logits = tf.transpose(logits_list, [1,0,2], name="all_logits")
        all_logits = tf.reshape(all_logits, [batch_size, self.maxlen])
        return all_logits

    

        


if __name__ == "__main__":
    vocab_size = 7621
    batch_size = 32
    d = Discriminator(vocab_size, batch_size = batch_size)
    #context_input = tf.get_variable("input_features", [batch_size, 196, 512])
    #sequence_input = tf.get_variable("sequence_input", [batch_size, 3, vocab_size])

    context_input = np.zeros((batch_size, 196, 512), dtype=np.float32)
    sequence_input = np.zeros((batch_size, 3, vocab_size), dtype=np.float32)
    #context_input = np.load("ims.npy")
    #sequence_input = np.load("triples.npy")
    logits = d.build_discriminator(context_input, sequence_input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print sess.run(output)
        print sess.run(logits)
