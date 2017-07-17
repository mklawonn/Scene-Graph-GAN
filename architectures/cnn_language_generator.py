import tensorflow as tf

class Generator(object):

    def __init__(self, vocab_size, dim_context=[196, 512], seq_len = 3, dim_hidden=512, batch_size=64):

        self.vocab_size = vocab_size
        self.dim_context = dim_context
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.noise_dim = dim_hidden
        self.flag_shape = dim_hidden

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        constant_initializer = tf.constant_initializer(0.05)

        self.noise_embed_W = tf.get_variable("noise_embed_W", [self.noise_dim, dim_hidden], initializer=he_initializer)
        self.noise_embed_b = tf.get_variable("noise_embed_b", [dim_hidden], initializer=constant_initializer)

        self.sequence_W = tf.get_variable("sequence_W", [self.flag_shape + self.noise_dim, dim_hidden*seq_len], initializer = he_initializer)
        self.sequence_b = tf.get_variable("sequence_b", [dim_hidden*seq_len], initializer = constant_initializer)

        self.conv_1_1 = tf.get_variable("conv_1_1", [3,dim_hidden,dim_hidden], initializer=he_initializer)
        self.conv_1_2 = tf.get_variable("conv_1_2", [3,dim_hidden,dim_hidden], initializer=he_initializer)

        self.conv_2_1 = tf.get_variable("conv_2_1", [3,dim_hidden,dim_hidden], initializer=he_initializer)
        self.conv_2_2 = tf.get_variable("conv_2_2", [3,dim_hidden,dim_hidden], initializer=he_initializer)

        self.conv_3_1 = tf.get_variable("conv_3_1", [3,dim_hidden,dim_hidden], initializer=he_initializer)
        self.conv_3_2 = tf.get_variable("conv_3_2", [3,dim_hidden,dim_hidden], initializer=he_initializer)

        self.conv_4_1 = tf.get_variable("conv_4_1", [3,dim_hidden,dim_hidden], initializer=he_initializer)
        self.conv_4_2 = tf.get_variable("conv_4_2", [3,dim_hidden,dim_hidden], initializer=he_initializer)

        self.conv_5_1 = tf.get_variable("conv_5_1", [3,dim_hidden,dim_hidden], initializer=he_initializer)
        self.conv_5_2 = tf.get_variable("conv_5_2", [3,dim_hidden,dim_hidden], initializer=he_initializer)

        self.output_conv = tf.get_variable("output_conv", [1,dim_hidden,vocab_size], initializer=he_initializer)

    def build_generator(self, context, batch_size, attributes_flag):
        noise = tf.random_uniform([batch_size, self.noise_dim])

        embedded_noise = tf.add(tf.matmul(noise, self.noise_embed_W), self.noise_embed_b)

        flag = tf.reshape(attributes_flag, [batch_size, 1])
        flag = tf.tile(flag, [1, self.flag_shape])

        #In order to generate a sequence from this noise and flag 
        #expand z_hat to be sequence_length*dim_context[1], then reshape
        flag_and_noise = tf.concat([embedded_noise, flag], axis=1)
        resnet_input = tf.add(tf.matmul(flag_and_noise, self.sequence_W), self.sequence_b)
        resnet_input = tf.reshape(resnet_input, [-1, self.dim_hidden, self.seq_len])

        #Begin resnet operations
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

        #Begin output generation, a 1x1 convolution producing vocab_size output
        output = tf.nn.conv1d(value = output, filters=self.output_conv, stride=1, padding='SAME', data_format='NCHW')
        #Rearrange into batch_size, seq_len, vocab_size output
        output = tf.transpose(output, [0, 2, 1])
        output = tf.nn.softmax(output)
        return output
