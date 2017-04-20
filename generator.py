import tensorflow as tf

class Generator(object):
    def __init__(self, vocab_size, lstm_cell_size=512, dim_context=512, maxlen=3, batch_size=64, context_shape=[196,512], bias_init_vector=None):
        self.vocab_size = vocab_size
        self.lstm_cell_size = lstm_cell_size
        self.dim_context = dim_context
        self.context_shape = context_shape
        self.maxlen = maxlen
        self.batch_size = batch_size

    def build_generator(self, context):
        #context = tf.placeholder("float32", [None, self.context_shape[0], self.context_shape[1]])

        #First permute the input randomly
        #Switch dim 0 and 1
        context = tf.transpose(context, perm=[1,0,2])
        #Shuffle along dim 0
        context = tf.random_shuffle(context)
        #Switch back
        context = tf.transpose(context, perm=[1,0,2])
        #TODO At test time won't you need to figure out which features ended up where?
        #keep track of the shuffling to do spatial correlation

        with tf.variable_scope("encoder"):
            #Then encode the input sequence
            #Declare sequence reader
            encoder = tf.contrib.rnn.LSTMCell(self.lstm_cell_size, use_peepholes=True)
            #The size of the attention window. Currently pays attention to image feats in thirds
            attn_length = int(self.context_shape[0] / 3)
            #TODO Understand math behind this attention model
            attention_wrapped_encoder = tf.contrib.rnn.AttentionCellWrapper(encoder, attn_length=attn_length, state_is_tuple=True)
            #Output is of shape [batch_size, max_len, cell.output_size]
            encoder_output, _ = tf.nn.dynamic_rnn(attention_wrapped_encoder, context, swap_memory=True, dtype=tf.float32)
            #Transpose to put time as first dim
            encoder_output = tf.transpose(encoder_output, perm=[1, 0, 2])

        with tf.variable_scope("decoder"):
            #Now decode
            decoder = tf.contrib.rnn.LSTMCell(self.lstm_cell_size, use_peepholes=True)
            attention_wrapped_decoder = tf.contrib.rnn.AttentionCellWrapper(decoder, attn_length=3, state_is_tuple=True)
            #Select the outputs from the encoder at regular "attn_length" long intervals
            feat_indices = [attn_length, 2*attn_length, self.context_shape[0] - 1]
            decoder_input = tf.gather(encoder_output, feat_indices)
            decoder_input = tf.transpose(decoder_input, perm=[1,0,2])
            #Output is of shape [batch_size, max_len, cell.output_size]
            decoder_output, _ = tf.nn.dynamic_rnn(attention_wrapped_decoder, decoder_input, dtype=tf.float32)
            #decoder_output = tf.transpose(decoder_output, perm=[1,0,2])

        
        with tf.variable_scope("classifier"):
            classifier_weights = tf.get_variable("proj_w", [self.lstm_cell_size, self.vocab_size], \
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            #classifier_weights_t = tf.transpose(classifier_proj_weights)
            classifier_weights = tf.reshape(classifier_weights, [1, self.lstm_cell_size, self.vocab_size])
            classifier_weights = tf.tile(classifier_weights, [self.batch_size, 1, 1])
            b = tf.get_variable("proj_b", [self.vocab_size])
            b = tf.reshape(b, [1, 1, self.vocab_size])
            b = tf.tile(b, [self.batch_size, self.maxlen, 1])
        
        logits = tf.add(tf.matmul(decoder_output, classifier_weights), b)
        #print(tf.nn.softmax(logits))
        #return tf.argmax(tf.nn.softmax(logits), axis=-1)
        return tf.nn.softmax(logits)

if __name__ == "__main__":
    g = Generator(10000)
    softmax = g.build_generator()
    print softmax.get_shape()
