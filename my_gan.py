import os, sys
sys.path.append(os.getcwd())

import time
import json
import random

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

from tqdm import tqdm


class SceneGraphWGAN(object):
    def __init__(self, batch_path, path_to_vocab_json, BATCH_SIZE=64):
        self.batch_path = batch_path
        self.batch_path += "/" if self.batch_path[-1] != "/" else ""
        self.path_to_vocab_json = path_to_vocab_json
        self.path_to_vocab_json += "/" if self.path_to_vocab_json != "/" else ""

        with open(path_to_vocab_json, "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.decoder = {y[0]:x for x, y in self.vocab.iteritems()}

        #Hyperparameters
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQ_LEN = 3
        self.LAMBDA = 10
        self.CRITIC_ITERS = 10
        self.DIM = 512
        self.ITERS = 100000

    def softmax(self, logits):
        return tf.reshape(
            tf.nn.softmax(
                tf.reshape(logits, [-1, self.vocab_size])
            ),
            tf.shape(logits)
        )

    def make_noise(self, shape):
        return tf.random_normal(shape)

    def ResBlock(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.1', self.DIM, self.DIM, 5, output)
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.2', self.DIM, self.DIM, 5, output)
        return inputs + (0.3*output)

    def Generator(self, n_samples, prev_outputs=None):
        noise_dim = 128
        output = self.make_noise(shape=[n_samples, noise_dim])
        output = lib.ops.linear.Linear('Generator.Input', noise_dim, self.SEQ_LEN*self.DIM, output)
        output = tf.reshape(output, [-1, self.DIM, self.SEQ_LEN])
        output = self.ResBlock('Generator.1', output)
        output = self.ResBlock('Generator.2', output)
        output = self.ResBlock('Generator.3', output)
        output = self.ResBlock('Generator.4', output)
        output = self.ResBlock('Generator.5', output)
        output = lib.ops.conv1d.Conv1D('Generator.Output', self.DIM, self.vocab_size, 1, output)
        output = tf.transpose(output, [0, 2, 1])
        output = self.softmax(output)
        return output

    def Discriminator(self, triple_input):
        output = tf.transpose(triple_input, [0,2,1])
        output = lib.ops.conv1d.Conv1D('Discriminator.Input', self.vocab_size, self.DIM, 1, output)
        output = self.ResBlock('Discriminator.1', output)
        output = self.ResBlock('Discriminator.2', output)
        output = self.ResBlock('Discriminator.3', output)
        output = self.ResBlock('Discriminator.4', output)
        output = self.ResBlock('Discriminator.5', output)
        output = tf.reshape(output, [-1, self.SEQ_LEN*self.DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', self.SEQ_LEN*self.DIM, 1, output)
        return output

    def Loss(self):
        #real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, self.SEQ_LEN])
        #real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
        self.real_inputs = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQ_LEN, self.vocab_size])
        fake_inputs = self.Generator(self.BATCH_SIZE)
        fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

        self.fake_inputs = fake_inputs

        disc_real = self.Discriminator(self.real_inputs) 
        disc_fake = self.Discriminator(fake_inputs)

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gen_cost = -tf.reduce_mean(disc_fake)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.BATCH_SIZE,1,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_inputs - self.real_inputs
        interpolates = self.real_inputs + (alpha*differences)
        gradients = tf.gradients(self.Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += self.LAMBDA*gradient_penalty

        self.disc_cost = disc_cost
        self.gen_cost = gen_cost

        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    def DataGenerator(self):
        filenames = ["{}{}".format(self.batch_path, i) for i in os.listdir(self.batch_path)]
        for f in filenames:
            npz = np.load(f)
            big_arr = npz['arr_0']
            all_pairs = []
            for i in xrange(0, big_arr.shape[0], 2):
                im_feats = big_arr[i]
                caps = big_arr[i+1]
                for c in xrange(caps.shape[0]):
                    all_pairs.append((im_feats, caps[c]))
            indices = list(range(len(all_pairs)))
            random.shuffle(indices)
            while len(indices) > self.BATCH_SIZE:
                im_batch = np.array([all_pairs[i][0] for i in indices[-self.BATCH_SIZE:]], dtype=np.float32)
                triple_batch = np.array([all_pairs[i][1] for i in indices[-self.BATCH_SIZE:]])
                t_batch = np.zeros((self.BATCH_SIZE, 3, self.vocab_size), dtype=np.float32)
                for row in range(t_batch.shape[0]):
                    for token in range(t_batch.shape[1]):
                        t_batch[row, token, triple_batch[row, token]] = 1.0
                del indices[-self.BATCH_SIZE:]
                yield im_batch, t_batch


    def Train(self, epochs):
        self.Loss()
        start_time = time.time()
        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            def generate_samples():
                samples = session.run(self.fake_inputs)
                samples = np.argmax(samples, axis=2)
                decoded_samples = []
                for i in xrange(len(samples)):
                    decoded = []
                    for j in xrange(len(samples[i])):
                        decoded.append(self.decoder[samples[i][j]])
                    decoded_samples.append(tuple(decoded))
                return decoded_samples

            gen = self.DataGenerator()

            iteration = 0
            for im_batch, triple_batch in self.DataGenerator():
                #Train Generator
                if iteration > 0:
                    _ = session.run(self.gen_train_op)

                #Train Critic
                for i in xrange(self.CRITIC_ITERS):
                    #im_batch, triple_batch = gen.next()
                    _disc_cost, _ = session.run(
                        [self.disc_cost, self.disc_train_op],
                        feed_dict={self.real_inputs:triple_batch}
                    )

                if iteration % 200 == 0:

                    stop_time = time.time()
                    duration = (stop_time - start_time) / 200.0
                    start_time = stop_time
                    _gen_cost = session.run(self.gen_cost)

                    print "Time {}/itr, Step: {}, generator loss: {}, discriminator loss: {}".format(
                            duration, iteration, _gen_cost, _disc_cost)

                if iteration % 5000 == 0:
                    samples = []
                    for i in xrange(10):
                        samples.extend(generate_samples())

                    with open('./samples/samples_{}.txt'.format(iteration), 'w') as f:
                        for s in samples:
                            s = " ".join(s)
                            f.write(s + "\n")
                iteration += 1


if __name__ == "__main__":
    arg_dict = {}
    with open("./config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            arg_dict[line_[0]] = line_[1]
    batch_path = "{}{}".format(arg_dict["visual_genome"], "batches")
    path_to_vocab_json = arg_dict["vocab"]
    logs_dir = arg_dict["logs"]
    BATCH_SIZE = 64
    wgan = SceneGraphWGAN(batch_path, path_to_vocab_json, BATCH_SIZE=BATCH_SIZE)
    #wgan.create_network()
    #wgan.initialize_network(logs_dir)
    #wgan.train_model(25)
    wgan.Train(32)
