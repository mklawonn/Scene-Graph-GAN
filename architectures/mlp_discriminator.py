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


        self.context_embed_W = tf.get_variable("context_embed_W", [dim_context, dim_hidden], initializer=he_initializer)
        self.context_embed_b = tf.get_variable("context_embed_b", [dim_hidden], initializer=constant_initializer)

        self.triple_embed_W = tf.get_variable("triple_embed_W", [seq_len*vocab_size, dim_hidden*2], initializer=he_initializer)
        self.triple_embed_b = tf.get_variable("triple_embed_b", [dim_hidden*2], initializer=constant_initializer)

        self.triple_embed_W2 = tf.get_variable("triple_embed_W2", [dim_hidden*2, dim_hidden], initializer=he_initializer)
        self.triple_embed_b2 = tf.get_variable("triple_embed_b2", [dim_hidden], initializer=constant_initializer)

        self.output_W = tf.get_variable("output_W", [dim_hidden*2, dim_hidden], initializer=he_initializer)
        self.output_b = tf.get_variable("output_b", [dim_hidden], initializer=constant_initializer)

        self.output_W2 = tf.get_variable("output_W2", [dim_hidden, 1], initializer=he_initializer)
        self.output_b2 = tf.get_variable("output_b2", [1], initializer=constant_initializer)

    def build_discriminator(self, context, input_triples):
        #Flatten the input triples
        input_triples = tf.reshape(input_triples, [self.batch_size, self.seq_len*self.vocab_size])
        
        embedded_triple = tf.add(tf.matmul(input_triples, self.triple_embed_W), self.triple_embed_b)
        embedded_triple = tf.nn.relu(embedded_triple)

        embedded_triple = tf.add(tf.matmul(embedded_triple, self.triple_embed_W2), self.triple_embed_b2)
        embedded_triple = tf.nn.relu(embedded_triple)

        embedded_context = tf.add(tf.matmul(context, self.context_embed_W), self.context_embed_b)
        embedded_context = tf.nn.relu(embedded_context)

        context_triple_concat = tf.concat([embedded_context, embedded_triple], axis=1)
        
        hidden_logits = tf.add(tf.matmul(context_triple_concat, self.output_W), self.output_b)
        hidden_logits = tf.nn.relu(hidden_logits)

        output_logits = tf.add(tf.matmul(hidden_logits, self.output_W2), self.output_b2)

        return output_logits
