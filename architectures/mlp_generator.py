import tensorflow as tf

class Generator(object):

    def __init__(self, vocab_size, dim_context=4096, seq_len = 3, dim_hidden=512, batch_size=64):

        self.vocab_size = vocab_size
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_dim = dim_context

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        constant_initializer = tf.constant_initializer(0.05)


        self.context_embed_W = tf.get_variable("context_embed_W", [dim_context*2, dim_hidden*2], initializer=he_initializer)
        self.context_embed_b = tf.get_variable("context_embed_b", [dim_hidden*2], initializer=constant_initializer)

        self.context_embed_W2 = tf.get_variable("context_embed_W2", [dim_hidden*2, dim_hidden], initializer=he_initializer)
        self.context_embed_b2 = tf.get_variable("context_embed_b2", [dim_hidden], initializer=constant_initializer)

        self.output_W = tf.get_variable("output_W", [dim_hidden, dim_hidden*2], initializer=he_initializer)
        self.output_b = tf.get_variable("output_b", [dim_hidden*2], initializer=constant_initializer)

        self.output_W2 = tf.get_variable("output_W2", [dim_hidden*2, dim_hidden*4], initializer=he_initializer)
        self.output_b2 = tf.get_variable("output_b2", [dim_hidden*4], initializer=constant_initializer)

        self.output_W3 = tf.get_variable("output_W3", [dim_hidden*4, seq_len*vocab_size], initializer=he_initializer)
        self.output_b3 = tf.get_variable("output_b3", [seq_len*vocab_size], initializer=constant_initializer)

    def build_generator(self, context):

        noise = tf.random_uniform([self.batch_size, self.noise_dim])

        context_noise_concatenated = tf.concat([context, noise], axis=1)
        
        embedded_context = tf.add(tf.matmul(context_noise_concatenated, self.context_embed_W), self.context_embed_b)
        embedded_context = tf.nn.relu(embedded_context)

        embedded_context = tf.add(tf.matmul(embedded_context, self.context_embed_W2), self.context_embed_b2)
        embedded_context = tf.nn.relu(embedded_context)

        hidden_logits = tf.add(tf.matmul(embedded_context, self.output_W), self.output_b)
        hidden_logits = tf.nn.relu(hidden_logits)

        hidden_logits = tf.add(tf.matmul(hidden_logits, self.output_W2), self.output_b2)
        hidden_logits = tf.nn.relu(hidden_logits)

        output_logits = tf.add(tf.matmul(hidden_logits, self.output_W3), self.output_b3)
        output = tf.reshape(output_logits, [self.batch_size, self.seq_len, self.vocab_size])
        output = tf.nn.softmax(output)

        return output
