import os
import tensorflow as tf
import numpy as np


class Discriminator(object):

    def __init__(self, vocab_size, dim_embed=512, dim_context=512, dim_hidden=512, n_lstm_steps=3, batch_size=64, context_shape=[196,512], bias_init_vector=None):
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.context_shape = context_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size
        self.maxlen = n_lstm_steps

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
        self.decode_word_W = tf.get_variable("decode_word_W", [dim_embed, 1], initializer=xavier_initializer)
        self.decode_word_b = tf.get_variable("decode_word_b", [1], initializer=constant_initializer)"""
        self.Wemb = tf.get_variable("Wemb", [vocab_size, dim_embed], initializer=xavier_initializer)
        self.embed_triple_W = tf.get_variable("embed_triple_W", [self.maxlen*dim_embed, dim_embed], initializer=xavier_initializer)
        self.embed_triple_b = tf.get_variable("embed_triple_b", [dim_embed], initializer=constant_initializer)

        self.embed_image_W = tf.get_variable("embed_image_W", [context_shape[0]*context_shape[1], dim_embed], initializer=xavier_initializer)
        self.embed_image_b = tf.get_variable("embed_image_b", [dim_embed], initializer=constant_initializer)

        self.out_W_1 = tf.get_variable("out_W_1", [dim_embed*2, dim_embed], initializer=xavier_initializer)
        self.out_b_1 = tf.get_variable("out_b_1", [dim_embed], initializer=constant_initializer)

        self.out_W_2 = tf.get_variable("out_W_2", [dim_embed, 1], initializer=xavier_initializer)
        self.out_b_2 = tf.get_variable("out_b_2", [1], initializer=constant_initializer)


    def get_initial_lstm(self, mean_context):
        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This is done via the mean_context ops
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    """def build_discriminator(self, context, triples):
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

            #Will produce a tensor of shape [batch_size, 1, word_embedding_size]
            seq_input_at_ind = tf.reshape(triples[:, ind, :], [self.batch_size, -1])
            word_emb = tf.matmul(seq_input_at_ind, self.Wemb)

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
            #word_prob = tf.nn.softmax(logit_words)
            #word_prediction = tf.argmax(logit_words, 1)
            #l.append(word_prob)

        #word_probs = tf.stack(l)
        #word_probs = tf.transpose(word_probs, [1, 0, 2])
        return logit_words"""

    
    """def build_discriminator(self, context, triples):
        #Embed the triple
        flattened_seq = tf.reshape(triples, [self.batch_size*self.maxlen, self.vocab_size])
        embedded_seq = tf.matmul(flattened_seq, self.Wemb)
        embedded_seq = tf.reshape(embedded_seq, [self.batch_size, self.maxlen, self.dim_embed])
        embedded_seq = tf.reshape(embedded_seq, [self.batch_size, self.maxlen*self.dim_embed])
        embedded_seq = tf.add(tf.matmul(embedded_seq, self.embed_triple_W), self.embed_triple_b)

        #Embed the image features
        flattened_im = tf.reshape(context, [self.batch_size, self.context_shape[0]*self.context_shape[1]])
        embedded_im = tf.add(tf.matmul(flattened_im, self.embed_image_W), self.embed_image_b)

        #Concatenate
        concatenated = tf.concat([embedded_seq, embedded_im], 1)

        #Run concatenated vector through output MLP
        h1 = tf.add(tf.matmul(concatenated, self.out_W_1), self.out_b_1)
        h2 = tf.add(tf.matmul(h1, self.out_W_2), self.out_b_2)
        return h2"""

    def ResBlock(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        
        output = lib.ops.conv1d.Conv1D(name+'.1', self.DIM, self.DIM, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.2', self.DIM, self.DIM, 5, output)
        return inputs + (0.3*output)


    def build_discriminator(self, context, triples):
        output = inputs
        output = tf.nn.relu(output)
        #output = lib.ops.conv1d.Conv1D(name+'.1', self.DIM, self.DIM, 5, output)
        output = tf.nn.conv1d(output, self.conv_1, stride=1, padding="VALID")
        output = tf.nn.relu(output)
        #output = lib.ops.conv1d.Conv1D(name+'.2', self.DIM, self.DIM, 5, output)
        output = tf.nn.conv1d(output, self.conv_2, stride=1, padding="VALID")
        return inputs + (0.3*output)


        


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
