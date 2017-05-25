""" Code adapted from Tensorflow Show Attend and Tell implementaion
    available here: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow
    Orignal paper found here: https://arxiv.org/pdf/1502.03044.pdf"""


import os
import tensorflow as tf
import numpy as np

class Generator(object):
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/np.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, vocab_size, dim_embed=512, dim_context=512, dim_hidden=512, n_lstm_steps=3, batch_size=64, context_shape=[196,512], bias_init_vector=None):
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.context_shape = context_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size
        self.maxlen = n_lstm_steps

        #Initialize the word embeddings
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([vocab_size, dim_embed], -1.0, 1.0), name='Wemb')

        #Initial hidden state of LSTM
        self.init_hidden_W = self.init_weight(dim_context, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

        #Initial memory of LSTM
        self.init_memory_W = self.init_weight(dim_context, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

        #Initialize LSTM Weights
        self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W')
        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')

        #Weights for image_encoding
        self.image_encode_W = self.init_weight(dim_context, dim_hidden*4, name='image_encode_W')

        #Initialize attention weights
        self.image_att_W = self.init_weight(dim_context, dim_context, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_context, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_context, name='pre_att_b')

        #I'm pretty sure this is the f_{att} model discussed in the paper
        #(the attention model conditioned on h_{t-1}
        self.att_W = self.init_weight(dim_context, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        #Initialize decoder weights
        #Goes from LSTM hidden state to a "word" embedding code
        self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')

        #Goes from the word embedding code to an output in the vocab_size space
        #(i.e slap a softmax on it and the output is a probability for each word in the vocab being
        #correct at that timestep)
        self.decode_word_W = self.init_weight(dim_embed, vocab_size, name='decode_word_W')

        if bias_init_vector is not None:
            self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(vocab_size, name='decode_word_b')


    def get_initial_lstm(self, mean_context):
        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This is done via the mean_context ops
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    def build_generator(self, context):
        #The context vector (\hat{z_t} in the paper) is a dynamic representation of a relevant part of an image 
        #at time t
        #Here however, the context has more dimensions. The batch_size is obvious, but the context shape refers
        #to the number of annotations and the dimensionality of each annotation
        #i.e self.context_shape[0] = #annotations (called L in the paper) and self.context_shape[1] = #dimensions
        #(D in the paper) The paper describes this in section 3.1.1
        #context = tf.placeholder("float32", [self.batch_size, self.context_shape[0], self.context_shape[1]])
        #The sentence represents the collection of words
        #sentence = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        #The mask is
        #mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This reduce mean call averages across the second dimension, i.e the L annotations 

        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))#(batch_size, D)

        #The flattened context
        #The tf.reshape function takes context and reshapes it so that the total number of elements
        #remains constant. This is done by the -1 argument. So the shape will be [(batch_size*L), D]
        context_flat = tf.reshape(context, [-1, self.dim_context])
        #Encode the context by multiplying by image attention weights
        context_encode = tf.matmul(context_flat, self.image_att_W) # (batch_size*L, D)
        context_encode = tf.reshape(context_encode, [-1, self.context_shape[0], self.context_shape[1]]) #(batch_size, L, D)

        #The number of lstm_steps is how many outputs the LSTM will make
        l = []
        for ind in range(self.n_lstm_steps):

            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    #word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1])
                    #The calculation will be made based on the last word looked at
                    word_emb = tf.nn.embedding_lookup(self.Wemb, word_prediction)

            #This is the T_{D+m+n,n} affine transformation referred to in section 3.1.2
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # (batch_size, hidden*4)
            
            context_encode = context_encode + \
                 tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
                 self.pre_att_b

            context_encode = tf.nn.tanh(context_encode)

            #context_encode: 3D -> flat required
            context_encode_flat = tf.reshape(context_encode, [-1, self.dim_context]) # (batch_size*L, D)
            alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b # (batch_size*L, 1)
            alpha = tf.reshape(alpha, [-1, self.context_shape[0]]) # (batch_size, L)
            #Alpha now represents the relative importance to give to each of the L annotations for 
            #generating the next word
            #TODO: Alter how the alphas are computed rather than dropping context?
            # maybe impose a penalty for this alpha being too similar to the last alpha?
            # would also have to change how it initializes then too
            alpha = tf.nn.softmax( alpha )

            #This is the Phi function for soft attention as explained in section 4.2
            #Thus weighted context is equal to \hat{z}
            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) #(batch_size, D)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
            #LSTM Computations
            #tf.split(
            i, f, o, new_c = tf.split(lstm_preactive, 4, 1)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, 0.5)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            word_prob = tf.nn.softmax(logit_words)
            word_prediction = tf.argmax(logit_words, 1)
            l.append(word_prob)

        word_probs = tf.stack(l)
        word_probs = tf.transpose(word_probs, [1, 0, 2])
        return word_probs

if __name__ == "__main__":
    g = Generator(10000)
    context = tf.Variable(tf.zeros([64, 196, 512]))
    softmax = g.build_generator(context)
    print softmax.get_shape()
