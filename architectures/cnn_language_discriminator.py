import tensorflow as tf

class Discriminator(object):

    def __init__(self, vocab_size, dim_context=[196, 512], seq_len = 3, dim_hidden=512, batch_size=64):

        self.vocab_size = vocab_size
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.flag_shape = dim_hidden

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        constant_initializer = tf.constant_initializer(0.05)


        #1D convolution with filter size of 1 and stride of 1 applied to the input triple
        #Produces dim_hidden output channels
        #TODO: Does it make sense to downsample channels like this?
        self.triple_embedder_1 = tf.get_variable("triple_embedder_1", [2,vocab_size,vocab_size], initializer=he_initializer)
        self.triple_embedder_2 = tf.get_variable("triple_embedder_2", [2,vocab_size,dim_hidden], initializer=he_initializer)
        self.triple_embedder_3 = tf.get_variable("triple_embedder_3", [2,dim_hidden,dim_hidden], initializer=he_initializer)

        #The dim_hidden*2 is the size of the embedded context
        self.att_W = tf.get_variable("att_W", [dim_hidden*4, dim_context[0]], initializer = he_initializer)
        self.att_b = tf.get_variable("att_b", [dim_context[0]], initializer = constant_initializer)

        #Resnet weights
        #3x3 Filters with a stride of 1
        self.conv_1_1 = tf.get_variable("conv_1_1", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)
        self.conv_1_2 = tf.get_variable("conv_1_2", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)

        self.conv_2_1 = tf.get_variable("conv_2_1", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)
        self.conv_2_2 = tf.get_variable("conv_2_2", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)

        self.conv_3_1 = tf.get_variable("conv_3_1", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)
        self.conv_3_2 = tf.get_variable("conv_3_2", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)

        self.conv_4_1 = tf.get_variable("conv_4_1", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)
        self.conv_4_2 = tf.get_variable("conv_4_2", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)

        self.conv_5_1 = tf.get_variable("conv_5_1", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)
        self.conv_5_2 = tf.get_variable("conv_5_2", [3,self.flag_shape+dim_hidden,self.flag_shape+dim_hidden], initializer=he_initializer)

        self.output_W_1 = tf.get_variable("output_W_1", [(self.flag_shape+dim_hidden)*seq_len, dim_hidden], initializer=he_initializer)
        self.output_b_1 = tf.get_variable("output_b_1", [dim_hidden], initializer=constant_initializer)

        self.output_W_2 = tf.get_variable("output_W_2", [dim_hidden, 1], initializer=he_initializer)
        self.output_b_2 = tf.get_variable("output_b_2", [1], initializer=constant_initializer)


        
    def build_discriminator(self, context, input_triples, batch_size, attributes_flag):
        #Switch to batch_size, vocab_size, seq_len in order to convolve over the vocab_size using NCHW
        embedded_triple = tf.transpose(input_triples, [0, 2, 1])
        embedded_triple = tf.nn.conv1d(value=embedded_triple, filters=self.triple_embedder_1, stride=1, padding='SAME', data_format='NCHW')
        embedded_triple = tf.nn.relu(embedded_triple)
        embedded_triple = tf.nn.conv1d(value=embedded_triple, filters=self.triple_embedder_2, stride=1, padding='SAME', data_format='NCHW')
        embedded_triple = tf.nn.relu(embedded_triple)
        embedded_triple = tf.nn.conv1d(value=embedded_triple, filters=self.triple_embedder_3, stride=1, padding='SAME', data_format='NCHW')

        flag = tf.reshape(attributes_flag, [batch_size, 1, 1])
        flag = tf.tile(flag, [1, self.flag_shape, self.seq_len])

        embedded_and_flag = tf.concat([flag, embedded_triple], axis=1)
        resnet_input = embedded_and_flag

        output = resnet_input
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_1_1, stride=1, padding='SAME', data_format='NCHW')
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_1_2, stride=1, padding='SAME', data_format='NCHW')
        resnet_input = resnet_input + (0.3*output)

        output = resnet_input
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_2_1, stride=1, padding='SAME', data_format='NCHW')
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_2_2, stride=1, padding='SAME', data_format='NCHW')
        resnet_input = resnet_input + (0.3*output)

        output = resnet_input
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_3_1, stride=1, padding='SAME', data_format='NCHW')
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_3_2, stride=1, padding='SAME', data_format='NCHW')
        resnet_input = resnet_input + (0.3*output)

        output = resnet_input
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_4_1, stride=1, padding='SAME', data_format='NCHW')
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_4_2, stride=1, padding='SAME', data_format='NCHW')
        resnet_input = resnet_input + (0.3*output)

        output = resnet_input
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_5_1, stride=1, padding='SAME', data_format='NCHW')
        output = tf.nn.relu(output)
        output = tf.nn.conv1d(value=output, filters=self.conv_5_2, stride=1, padding='SAME', data_format='NCHW')
        output = resnet_input + (0.3*output)

        #Output layer to produce logits
        output = tf.reshape(output, [-1, (self.flag_shape+self.dim_hidden)*self.seq_len])
        output = tf.add(tf.matmul(output, self.output_W_1), self.output_b_1)
        output = tf.nn.relu(output)
        output_logits = tf.add(tf.matmul(output, self.output_W_2), self.output_b_2)

        return output_logits
