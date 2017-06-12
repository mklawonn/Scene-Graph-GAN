import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self, vocab_size, context_shape=[196, 512], seq_len = 3, dim_hidden=512, batch_size=64):
        self.vocab_size = vocab_size
        self.context_shape = context_shape
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_dim = context_shape[1]
        self.dim_embed = 300
        self.soft_gumbel_temp = 0.9

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        constant_initializer = tf.constant_initializer(0.05)

        #The weights for encoding the context to feed it into the attention MLP
        #Needs to be encoded in order to combine it with the previous hidden state
        self.context_encode_W = tf.get_variable("context_encode", [self.context_shape[0]*self.context_shape[1], self.context_shape[1]*2], initializer=xavier_initializer)

        self.Wemb = tf.get_variable("Wemb", [self.vocab_size, self.dim_embed], initializer=xavier_initializer)

        #self.att_W = tf.get_variable("att_W", [(self.dim_context*2)+self.dim_hidden+self.noise_dim, self.context_shape[0]], initializer=xavier_initializer)
        self.att_W = tf.get_variable("att_W", [(self.context_shape[1]*2)+self.dim_hidden+self.noise_dim, self.context_shape[0]], initializer=xavier_initializer)
        self.att_b = tf.get_variable("att_b", [self.context_shape[0]], initializer=constant_initializer)

        self.lstm_W = tf.get_variable("lstm_W", [self.context_shape[1]+self.dim_hidden+self.dim_embed, self.dim_hidden*4], initializer=xavier_initializer)
        self.lstm_b = tf.get_variable("lstm_b", [self.dim_hidden*4], initializer=constant_initializer)

        self.init_hidden_W = tf.get_variable("init_hidden_W", [self.context_shape[1], self.dim_hidden], initializer=xavier_initializer)
        self.init_hidden_b = tf.get_variable("init_hidden_b", [self.dim_hidden], initializer=constant_initializer)

        self.init_memory_W = tf.get_variable("init_memory_W", [self.context_shape[1], self.dim_hidden], initializer=xavier_initializer)
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


    def build_generator(self, context):
        flattened_context = tf.reshape(context, [self.batch_size, self.context_shape[0]*self.context_shape[1]])
        encoded_context = tf.matmul(flattened_context, self.context_encode_W)

        noise = tf.random_uniform([self.batch_size, self.noise_dim])
        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))#(batch_size, dim_hidden)

        l = []

        for i in range(self.seq_len):
            if i == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                #word_emb = tf.nn.embedding_lookup(self.Wemb, word_prediction)
                word_emb = tf.matmul(word_prob, self.Wemb)


            context_hidden_state_and_noise = tf.concat([encoded_context, h, noise], 1)
            e = tf.add(tf.matmul(context_hidden_state_and_noise, self.att_W), self.att_b)

            alpha = tf.nn.softmax(e)
            #alpha = tf.contrib.distributions.RelaxedOneHotCategorical(self.soft_gumbel_temp, logits=e).sample()
            z_hat = tf.reduce_sum(tf.multiply(context, tf.expand_dims(alpha, 2)), axis = 1) #Output is [batch size, D]
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

            #Could make the decoding a "deep output layer" by adding another layer
            logits = tf.add(tf.matmul(h, self.decode_lstm_W), self.decode_lstm_b)
            #word_prob = tf.nn.softmax(logits)
            #TODO Look at boundary GAN paper
            word_prob = tf.contrib.distributions.RelaxedOneHotCategorical(self.soft_gumbel_temp, logits=logits).sample()
            #word_prediction = tf.argmax(logits, 1)
            l.append(word_prob)


        word_probs = tf.stack(l)
        #We want it in shape [batch_size, seq_len, vocab_size]
        word_probs = tf.transpose(word_probs, [1,0,2])
        return word_probs


if __name__ == "__main__":
    batch_size = 64
    context = np.zeros((batch_size, 196, 512), dtype=np.float32)
    g = Generator(8000)
    print g.build_generator(context)
