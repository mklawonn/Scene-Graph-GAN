""" Code adapted from Tensorflow Show Attend and Tell implementaion
    available here: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow
    Orignal paper found here: https://arxiv.org/pdf/1502.03044.pdf"""


import os
import tensorflow as tf
import numpy as np

class Generator(object):
    """def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/np.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)"""

    def __init__(self, vocab_size, dim_embed=512, dim_context=512, dim_hidden=512, n_lstm_steps=3, batch_size=64, context_shape=[196,512], bias_init_vector=None):
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.context_shape = context_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size
        self.maxlen = n_lstm_steps
        #self.noise_dim = 128
        self.soft_gumbel_temp = 2.5

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        constant_initializer = tf.constant_initializer(0.05)
        """#Initialize the word embeddings
        self.Wemb = tf.get_variable("Wemb", [vocab_size, dim_embed], initializer=xavier_initializer)

        #Initial hidden state of LSTM
        self.init_hidden_W = tf.get_variable("init_hidden_W", [dim_context, dim_hidden], initializer=xavier_initializer)
        self.init_hidden_b = tf.get_variable("init_hidden_b", [dim_hidden], initializer=constant_initializer)

        #Initial memory of LSTM
        self.init_memory_W = tf.get_variable("init_memory_W", [dim_context, dim_hidden], initializer=xavier_initializer)
        self.init_memory_b = tf.get_variable("init_memory_b", [dim_hidden], initializer=constant_initializer)

        #Initialize LSTM Weights
        self.lstm_W = tf.get_variable("lstm_W", [dim_embed, dim_hidden*4], initializer=xavier_initializer)
        self.lstm_U = tf.get_variable("lstm_U", [dim_hidden, dim_hidden*4], initializer=xavier_initializer)
        self.lstm_b = tf.get_variable("lstm_b", [dim_hidden*4], initializer=constant_initializer)

        #Weights for image_encoding
        self.image_encode_W = tf.get_variable("image_encode_W", [dim_context, dim_hidden*4], initializer=xavier_initializer)

        #Initialize attention weights
        self.image_att_W = tf.get_variable("image_att_W", [dim_context, dim_context], initializer=xavier_initializer)
        self.hidden_att_W = tf.get_variable("hidden_att_W", [dim_hidden, dim_context], initializer=xavier_initializer)
        self.pre_att_b = tf.get_variable("pre_att_b", [dim_context], initializer=constant_initializer)

        #I'm pretty sure this is the f_{att} model discussed in the paper
        #(the attention model conditioned on h_{t-1}
        self.att_W = tf.get_variable("att_W", [dim_context, 1], initializer=xavier_initializer)
        self.att_b = tf.get_variable("att_b", [1], initializer=constant_initializer)

        #Initialize decoder weights
        #Goes from LSTM hidden state to a "word" embedding code
        self.decode_lstm_W = tf.get_variable("decode_lstm_W", [dim_hidden, dim_embed], initializer=xavier_initializer)
        self.decode_lstm_b = tf.get_variable("decode_lstm_b", [dim_embed], initializer=constant_initializer)

        #Goes from the word embedding code to an output in the vocab_size space
        #(i.e slap a softmax on it and the output is a probability for each word in the vocab being
        #correct at that timestep)
        self.decode_word_W = tf.get_variable("decode_word_W", [dim_embed, vocab_size], initializer=xavier_initializer)
        self.decode_word_b = tf.get_variable("decode_word_b", [vocab_size], initializer=constant_initializer)"""

        #The weights for encoding the context to feed it into the attention MLP
        #Needs to be encoded in order to combine it with the previous hidden state
        self.context_encode_W = tf.get_variable("context_encode", [self.context_shape[0]*self.context_shape[1], self.dim_context*2], initializer=xavier_initializer)

        self.Wemb = tf.get_variable("Wemb", [self.vocab_size, self.dim_embed], initializer=xavier_initializer)

        #self.att_W = tf.get_variable("att_W", [(self.dim_context*2)+self.dim_hidden+self.noise_dim, self.context_shape[0]], initializer=xavier_initializer)
        self.att_W = tf.get_variable("att_W", [(self.dim_context*2)+self.dim_hidden, self.context_shape[0]], initializer=xavier_initializer)
        self.att_b = tf.get_variable("att_b", [self.context_shape[0]], initializer=constant_initializer)

        self.lstm_W = tf.get_variable("lstm_W", [self.dim_context+self.dim_hidden+self.dim_embed, self.dim_hidden*4], initializer=xavier_initializer)
        self.lstm_b = tf.get_variable("lstm_b", [self.dim_hidden*4], initializer=constant_initializer)

        self.init_hidden_W = tf.get_variable("init_hidden_W", [self.dim_context, self.dim_hidden], initializer=xavier_initializer)
        self.init_hidden_b = tf.get_variable("init_hidden_b", [self.dim_hidden], initializer=constant_initializer)

        self.init_memory_W = tf.get_variable("init_memory_W", [self.dim_context, self.dim_hidden], initializer=xavier_initializer)
        self.init_memory_b = tf.get_variable("init_memory_b", [self.dim_hidden], initializer=constant_initializer)

        self.decode_lstm_W = tf.get_variable("decode_lstm_W", [self.dim_hidden, self.vocab_size], initializer=xavier_initializer)
        self.decode_lstm_b = tf.get_variable("decode_lstm_b", [self.vocab_size], initializer=constant_initializer)



    def get_initial_lstm(self, mean_context):
        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This is done via the mean_context ops
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    """def build_generator(self, context):
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
            #TODO: Alter how the alphas are computed
            #My current idea is to create a random masking vector the same shape as the alpha
            #vector. Values in this noise vector should sum to one. Then a term must be included in the
            #loss to penalize the attention model for ignoring the random mask. This term should penalize
            #the produced alphas if they are too far from the mask. Something like:
            #sum_{i=1}^{L}sum_{c=1}^{3}(alpha_{ic} - randomMask_{i}
            #Other idea: maybe impose a penalty for this alpha being too similar to the last alpha?
            #would also have to change how it initializes then too
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
        return word_probs"""

    def build_generator(self, context):
        #From the paper: The initial memory state and hidden state of the LSTM are predicted
        #by an average of the annotation vectors fed through two separate MLPs
        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))#(batch_size, dim_hidden)

        
        #noise = tf.random_normal([self.batch_size, self.noise_dim])

        l = []

        for ind in range(self.n_lstm_steps):
            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                word_emb = tf.nn.embedding_lookup(self.Wemb, word_prediction)
            #Calculate \hat{z}
            #Equation (4)
            #Concatenate flattened context with previous hidden state and the noise vector, and feed through attention mlp
            flattened_context = tf.reshape(context, [self.batch_size, self.context_shape[0]*self.context_shape[1]])
            encoded_context = tf.matmul(flattened_context, self.context_encode_W)
            #context_and_hidden_state = tf.concat([encoded_context, h, noise], 1)
            context_and_hidden_state = tf.concat([encoded_context, h], 1)
            e = tf.add(tf.matmul(context_and_hidden_state, self.att_W), self.att_b)
            #Equation (5)
            #alpha = tf.nn.softmax(e)
            alpha = tf.contrib.distributions.RelaxedOneHotCategorical(self.soft_gumbel_temp, logits=e).sample()
            #Equation (6) and (13): Soft attention model
            z_hat = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) #Output is [batch size, D]
            #Equation (1)
            #Concatenate \hat{z}_t , h_{t-1}, and the embedding of the previous word 
            lstm_input = tf.concat([z_hat, h, word_emb], 1)
            #Perform affine transformation of concatenated vector
            affine = tf.add(tf.matmul(lstm_input, self.lstm_W), self.lstm_b)
            i, f, o, g = tf.split(affine, 4, 1)
            #i_t = sigmoid(affine)
            i = tf.nn.sigmoid(i)
            #f_t = sigmoid(affine)
            f = tf.nn.sigmoid(f)
            #o_t = sigmoid(affine)
            o = tf.nn.sigmoid(o)
            #g_t = tanh(affine)
            g = tf.nn.tanh(g)

            #Equation (2)
            #c_t = f_t * c_{t-1} + i_t * g_t
            c = f * c + i * g
            #h_t = o_t * tanh(c_t)
            h = o * tf.nn.tanh(c)

            logits = tf.add(tf.matmul(h, self.decode_lstm_W), self.decode_lstm_b)
            word_prob = tf.nn.softmax(logits)
            word_prediction = tf.argmax(logits, 1)
            l.append(word_prob)

        word_probs = tf.stack(l)
        #We want it in shape [batch_size, seq_len, vocab_size]
        word_probs = tf.transpose(word_probs, [1,0,2])
        return word_probs


if __name__ == "__main__":
    g = Generator(10000)
    context = tf.Variable(tf.zeros([64, 196, 512]))
    softmax = g.build_generator(context)
    print softmax.get_shape()
