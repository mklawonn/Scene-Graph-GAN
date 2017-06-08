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


        #Embed the input
        self.context_embed_W = tf.get_variable("context_embed_W", [dim_context+self.noise_dim, dim_hidden*2], initializer=he_initializer)
        self.context_embed_b = tf.get_variable("context_embed_b", [dim_hidden*2], initializer=constant_initializer)

        self.context_embed_W2 = tf.get_variable("context_embed_W2", [dim_hidden*2, seq_len*dim_hidden], initializer=he_initializer)
        self.context_embed_b2 = tf.get_variable("context_embed_b2", [seq_len*dim_hidden], initializer=constant_initializer)

        #3x3 Filters with a stride of 1
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

        
    def ResBlock(self, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.1', self.DIM, self.DIM, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.2', self.DIM, self.DIM, 5, output)
        return inputs + (0.3*output)

    def Generator(self, n_samples, image_feats, prev_outputs=None):
        noise_dim = 1024
        #output = self.make_noise(shape=[n_samples, noise_dim])
        noise = self.make_noise(shape=[n_samples, noise_dim])
        output = tf.concat([image_feats, noise], axis=1)
        output = lib.ops.linear.Linear('Generator.Input', noise_dim+self.image_feat_dim, self.seq_len*self.DIM, output)
        output = tf.reshape(output, [-1, self.DIM, self.seq_len])
        output = self.ResBlock('Generator.1', output)
        output = self.ResBlock('Generator.2', output)
        output = self.ResBlock('Generator.3', output)
        output = self.ResBlock('Generator.4', output)
        output = self.ResBlock('Generator.5', output)
        output = lib.ops.conv1d.Conv1D('Generator.Output', self.DIM, self.vocab_size, 1, output)
        output = tf.transpose(output, [0, 2, 1])
        output = self.softmax(output)
        return output


    def build_generator(self, context):
        noise = tf.random_uniform([self.batch_size, self.noise_dim])

        context_noise_concatenated = tf.concat([context, noise], axis=1)

        context_encode = tf.add(tf.matmul(context_noise_concatenated, self.context_embed_W), self.context_embed_b)
        #context_encode = tf.nn.relu(context_encode)

        context_encode = tf.add(tf.matmul(context_encode, self.context_embed_W2), self.context_embed_b2)
        #context_encode = tf.nn.relu(context_encode)

        resnet_input = tf.reshape(context_encode, [-1, self.dim_hidden, self.seq_len])
        

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
        output = self.softmax(output)
        return output
