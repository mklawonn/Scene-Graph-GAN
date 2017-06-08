import tensorflow as tf

class Discriminator(object):

    def __init__(self, vocab_size, dim_context=4096, seq_len = 3, dim_hidden=512, batch_size=64):

        self.vocab_size = vocab_size
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        constant_initializer = tf.constant_initializer(0.05)

        #1D convolution with filter size of 1 and stride of 1 applied to the produced triple
        self.triple_embedder = tf.get_variable("output_conv", [1,dim_hidden,vocab_size], initializer=he_initializer)

        
    def build_discriminator(self, context, input_triples):
        #Switch to batch_size, vocab_size, seq_len in order to convolve over the vocab_size using NCHW
        embedded_triple = tf.transpose(input_triples, [0, 2, 1])
        embedded_triple = tf.nn.conv1d(value=embedded_triple, filters=self.triple_embedder, stride=1, padding='SAME', data_format='NCHW')

        embedded_context = tf.
       
        return output_logits
