import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self, vocab_size, context_shape=[196, 512], seq_len = 3, dim_hidden=512, dim_embed = 300, batch_size=64):
        self.vocab_size = vocab_size
        self.context_shape = context_shape
        self.dim_hidden = dim_hidden
        #self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_dim = context_shape[1]
        self.dim_embed = dim_embed
        self.soft_gumbel_temp = 0.9
        self.flag_shape = 512

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        constant_initializer = tf.constant_initializer(0.05)
        alpha_initializer = tf.constant_initializer(1)
        beta_initializer = tf.constant_initializer(0)

        #self.noise_emb_W = tf.get_variable("noise_emb_W", [self.noise_dim, self.noise_dim], initializer=xavier_initializer)
        #self.noise_emb_b = tf.get_variable("noise_emb_b", [self.noise_dim], initializer=constant_initializer)

        self.Wemb = tf.get_variable("Wemb", [self.vocab_size, self.dim_embed], initializer=xavier_initializer)

        self.lstm_W = tf.get_variable("lstm_W", [self.dim_hidden+self.dim_embed+self.noise_dim+self.flag_shape, self.dim_hidden*4], initializer=xavier_initializer)
        self.lstm_b = tf.get_variable("lstm_b", [self.dim_hidden*4], initializer=constant_initializer)

        self.init_hidden_W = tf.get_variable("init_hidden_W", [self.noise_dim, self.dim_hidden], initializer=xavier_initializer)
        self.init_hidden_b = tf.get_variable("init_hidden_b", [self.dim_hidden], initializer=constant_initializer)

        self.init_memory_W = tf.get_variable("init_memory_W", [self.noise_dim, self.dim_hidden], initializer=xavier_initializer)
        self.init_memory_b = tf.get_variable("init_memory_b", [self.dim_hidden], initializer=constant_initializer)

        self.decode_lstm_W = tf.get_variable("decode_lstm_W", [self.dim_hidden, self.vocab_size], initializer=xavier_initializer)
        self.decode_lstm_b = tf.get_variable("decode_lstm_b", [self.vocab_size], initializer=constant_initializer)

        #For layer normalization:
        self.init_hidden_alpha = tf.get_variable("init_hidden/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.init_hidden_beta = tf.get_variable("init_hidden/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.init_memory_alpha = tf.get_variable("init_memory/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.init_memory_beta = tf.get_variable("init_memory/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.i_alpha = tf.get_variable("i/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.i_beta = tf.get_variable("i/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.f_alpha = tf.get_variable("f/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.f_beta = tf.get_variable("f/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.o_alpha = tf.get_variable("o/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.o_beta = tf.get_variable("o/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

        self.g_alpha = tf.get_variable("g/LN/alpha", shape=[self.dim_hidden], initializer = alpha_initializer)
        self.g_beta = tf.get_variable("g/LN/beta", shape=[self.dim_hidden], initializer = beta_initializer)

    def get_initial_lstm(self, noise):
        #From the paper: "The initial memory state and hidden state of the LSTM are predicted by an average
        #of the annotation vectors fed through two separate MLPs (init,c and init,h)"
        #This is done via the mean_context ops
        #context_and_noise = tf.concat([mean_context, noise], 1)
        #initial_hidden = tf.nn.tanh(tf.add(tf.matmul(context_and_noise, self.init_hidden_W), self.init_hidden_b))
        #initial_memory = tf.nn.tanh(tf.add(tf.matmul(context_and_noise, self.init_memory_W), self.init_memory_b))

        initial_hidden = tf.add(tf.matmul(noise, self.init_hidden_W), self.init_hidden_b)
        initial_hidden = self.layer_normalization(initial_hidden, self.init_hidden_alpha, self.init_hidden_beta)
        initial_hidden = tf.nn.tanh(initial_hidden)
        initial_memory = tf.add(tf.matmul(noise, self.init_memory_W), self.init_memory_b)
        initial_memory = self.layer_normalization(initial_memory, self.init_memory_alpha, self.init_memory_beta)
        initial_memory = tf.nn.tanh(initial_memory)

        return initial_hidden, initial_memory

    def layer_normalization(self, inputs, scale, shift, epsilon = 1e-5):
        mean, var = tf.nn.moments(inputs, [1], keep_dims = True)

        normalized = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
        return normalized

    def build_generator(self, context, batch_size, attributes_flag):
        flag = tf.reshape(attributes_flag, [1, 1])
        flag = tf.tile(flag, [batch_size, self.flag_shape])

        #Does different noise make a difference?
        noise = tf.random_normal([batch_size, self.noise_dim], mean=0.0, stddev=1.0)

        h, c = self.get_initial_lstm(noise)#(batch_size, dim_hidden)

        l = []

        for i in range(self.seq_len):
            if i == 0:
                word_emb = tf.zeros([batch_size, self.dim_embed])
            else:
                word_emb = tf.nn.embedding_lookup(self.Wemb, word_prediction)

            lstm_input = tf.concat([h, word_emb, noise, flag], 1)
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

            #Could make the decoding a "deep output layer" by adding another layer
            logits = tf.add(tf.matmul(h, self.decode_lstm_W), self.decode_lstm_b)
            word_prob = tf.nn.softmax(logits)
            word_prediction = tf.argmax(word_prob, 1)
            l.append(word_prob)


        word_probs = tf.stack(l)
        #We want it in shape [batch_size, seq_len, vocab_size]
        word_probs = tf.transpose(word_probs, [1,0,2], "generator_output")
        return word_probs


if __name__ == "__main__":
    batch_size = 64
    context = np.zeros((batch_size, 196, 512), dtype=np.float32)
    g = Generator(8000)
    print g.build_generator(context)
