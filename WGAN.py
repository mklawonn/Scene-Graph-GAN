"""Adapted from https://github.com/shekkizh/WassersteinGAN.tensorflow"""


from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os, sys, inspect
import time
import json
#import utils
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator

class WGAN(object):
    def __init__(self, batch_path, path_to_vocab_json, batch_size=64, dim_generator_hidden=512, dim_embed=300, \
                 critic_iterations=10, clip_values=(-0.01, 0.01)):
        self.batch_size = batch_size
        self.dim_generator_hidden = dim_generator_hidden
        self.dim_embed = dim_embed
        self.batch_path = batch_path
        self.batch_path += "/" if self.batch_path[-1] != "/" else ""
        self.path_to_vocab_json = path_to_vocab_json
        self.path_to_vocab_json += "/" if self.path_to_vocab_json != "/" else ""
        with open(path_to_vocab_json, "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        print(self.vocab_size)
        self.decode = {y[0]:x for x,y in self.vocab.iteritems()}

        #Placeholders
        self.image_feats = tf.placeholder(tf.float32, [None, 196, 512])
        self.real_graphs = tf.placeholder(tf.float32, [None, 3, self.vocab_size])

        #self.gen_graphs = tf.placeholder(tf.int64, [None, 3])

        #WGAN Specific
        self.critic_iterations = critic_iterations
        #Rather than clipping, penalize the norm of the gradient of the critic with respect to its input
        #This is from updated recommendations in https://arxiv.org/pdf/1704.00028.pdf
        #self.clip_values = clip_values
        #Lambda is the coefficient determining the severity of the penalty (a hyperparameter)
        self.lambda_ = 10
        #Alpha is the lipschitz penalty
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)
        random.seed(42)
        

    #Python generator that returns training examples from files in the batch_path
    def _read_input(self):
        filenames = ["{}{}".format(self.batch_path, i) for i in os.listdir(self.batch_path)]
        for f in filenames:
            npz = np.load(f)
            big_arr = npz['arr_0']
            all_pairs = []
            for i in xrange(0, big_arr.shape[0], 2):
                im_feats = big_arr[i]
                caps = big_arr[i+1]
                #print(caps.shape)
                for c in xrange(caps.shape[0]):
                    all_pairs.append((im_feats, caps[c]))
            indices = list(range(len(all_pairs)))
            random.shuffle(indices)
            while len(indices) > self.batch_size:
                im_batch = np.array([all_pairs[i][0] for i in indices[-self.batch_size:]], dtype=np.float32)
                triple_batch = np.array([all_pairs[i][1] for i in indices[-self.batch_size:]])
                #print(triple_batch.shape)
                #print(big_arr.shape)
                #Convert to one hot
                t_batch = np.zeros((self.batch_size, 3, self.vocab_size), dtype=np.float32)
                for row in range(t_batch.shape[0]):
                    for token in range(t_batch.shape[1]):
                        t_batch[row, token, triple_batch[row, token]] = 1.0
                #t_batch = np.eye(self.vocab_size)[triple_batch]
                del indices[-self.batch_size:]
                yield im_batch, t_batch
            

    def _generator(self, image_feats, scope_name="generator"):
        maxlen = 3
        #image_size = self.resized_image_size // (2 ** (N - 1))
        with tf.variable_scope(scope_name) as scope:
            #(self, vocab_size, lstm_cell_size=512, dim_context=512, maxlen=3, batch_size=128, context_shape=[196,512], bias_init_vector=None)
            g = Generator(self.vocab_size, n_lstm_steps = maxlen, batch_size = self.batch_size)

            generated_words = g.build_generator(image_feats)
            return generated_words



    def _discriminator(self, image_feats, graphs, scope_name="discriminator", scope_reuse=False):
        maxlen = 3
        with tf.variable_scope(scope_name, reuse=scope_reuse) as scope:
            d = Discriminator(self.vocab_size, batch_size = self.batch_size)

            logits = d.build_discriminator(image_feats, graphs)
            return logits

    """def _cross_entropy_loss(self, logits, labels, name="x_entropy"):
        xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))
        tf.scalar_summary(name, xentropy)
        return xentropy"""

    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        #for grad, var in grads:
        #    #utils.add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    """def _setup_placeholder(self):
        #self.train_phase = tf.placeholder(tf.bool)
        #self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")"""

    def _gan_loss(self, logits_real, logits_fake, real_inputs, fake_inputs):
        self.discriminator_loss = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        self.logits_fake = logits_fake

        self.gen_loss = -tf.reduce_mean(logits_fake)

        #Lipschitz constraints for the new and improved WGAN training
        #TODO Figure out why this is causing the thing to break
        #Current theory is calling tf.gradients is causing the optimizer.compute_gradients to compute second derivatives
        #TODO figure out how they use variable scopes, because I think that's what's causing the issue
        #TODO tf.stop_gradients()
        """differences = fake_inputs - real_inputs
        interpolates = real_inputs + (self.alpha*differences)
        logits_interpolates = self._discriminator(self.image_feats, interpolates, scope_name="discriminator", scope_reuse=True)
        self.logits_interpolates = logits_interpolates
        gradients = tf.gradients(logits_interpolates, [interpolates])[0]
        self.gs = gradients
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.discriminator_loss += self.lambda_*gradient_penalty"""

        tf.summary.scalar("Discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("Generator_loss", self.gen_loss)



    def create_network(self, optimizer="Adam", learning_rate=1e-4,
                       optimizer_param=0.5):
        print("Setting up model...")
        self.gen_graphs = self._generator(self.image_feats, scope_name="generator")

        #tf.image_summary("image_real", self.images, max_images=2)
        #tf.image_summary("image_generated", self.gen_images, max_images=2)

        #self.image_feats is a placeholder for the current batch of image feats
        #self.real_graphs is a placeholder for the current batch of actual scene graphs
        logits_real = self._discriminator(self.image_feats, self.real_graphs, scope_name="discriminator")

        #self.image_feats is a placeholder for the current batch of image feats
        #self.gen_graphs is the current batch of generated scene graphs
        logits_fake = self._discriminator(self.image_feats, self.gen_graphs, scope_name="discriminator", scope_reuse=True)

        # utils.add_activation_summary(tf.identity(discriminator_real_prob, name='disc_real_prob'))
        # utils.add_activation_summary(tf.identity(discriminator_fake_prob, name='disc_fake_prob'))

        # Loss calculation
        self._gan_loss(logits_real, logits_fake, self.real_graphs, self.gen_graphs)

        train_variables = tf.trainable_variables()

        """for v in train_variables:
            # print (v.op.name)
            #utils.add_to_regularization_and_summary(var=v)"""

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        # print(map(lambda x: x.op.name, generator_variables))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        # print(map(lambda x: x.op.name, discriminator_variables))

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)

        #TODO Figure out why it has to be in this order ...
        self.generator_train_op = self._train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, optim)

    def initialize_network(self, logs_dir):
        print("Initializing network...")
        self.logs_dir = logs_dir
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        #self.summary_writer = tf.train.SummaryWriter(self.logs_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #TODO Temporarily disabled to debug
            return
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    def train_model(self, epochs):
        print("Training Wasserstein GAN model...")
        #Using the weight normalization rather than gradient clipping now
        #clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
        #                             var in self.discriminator_variables]

        start_time = time.time()

        """def get_feed_dict(train_phase=True):
            batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
            feed_dict = {self.z_vec: batch_z, self.train_phase: train_phase}
            return feed_dict"""

        #for itr in tqdm(xrange(1, max_iterations)):

        DEBUG = True
        for epoch in tqdm(range(epochs)):
            itr = 0
            for im_batch, triple_batch in self._read_input():
                if itr < 25 or itr % 500 == 0:
                    critic_itrs = 25
                else:
                    critic_itrs = self.critic_iterations

                feed_dict = {self.image_feats : im_batch, self.real_graphs : triple_batch}

                for critic_itr in range(critic_itrs):
                    self.sess.run(self.discriminator_train_op, feed_dict=feed_dict)
                    print(self.sess.run(self.discriminator_loss, feed_dict=feed_dict))
                    #self.sess.run(clip_discriminator_var_op)

                if True:
                    sys.exit(1)

                self.sess.run(self.generator_train_op, feed_dict=feed_dict)

                if itr % 100 == 0:
                    #summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    #self.summary_writer.add_summary(summary_str, itr)
                    pass

                if itr % 200 == 0:
                    stop_time = time.time()
                    duration = (stop_time - start_time) / 200.0
                    start_time = stop_time
                    g_loss_val, d_loss_val = self.sess.run([self.gen_loss, self.discriminator_loss],
                                                           feed_dict=feed_dict)
                    print("Time: %g/itr, Step: %d, generator loss: %g, discriminator_loss: %g" % (
                        duration, itr, g_loss_val, d_loss_val))

                if itr % 5000 == 0:
                    self.saver.save(self.sess, self.logs_dir + "model.ckpt", global_step=itr)
                    self.visualize_model(feed_dict)
                itr += 1


    def visualize_model(self, feed_dict):
        print("-"*50)
        print("Sampling graphs from model...")
        #batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        #feed_dict = {self.z_vec: batch_z, self.train_phase: False}

        #images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        captions = self.sess.run(self.gen_graphs, feed_dict = feed_dict)
        argmax_ = np.argmax(captions, axis=2)
        for i in range(argmax_.shape[0]):
            s = "{} {} {}".format(self.decode[argmax_[i][0]], self.decode[argmax_[i][1]], self.decode[argmax_[i][2]])
            print(s)
        print("-"*50)
        #images = utils.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        #shape = [4, self.batch_size // 4]
        #utils.save_imshow_grid(images, self.logs_dir, "generated.png", shape=shape)

if __name__ == "__main__":
    arg_dict = {}
    with open("./config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            arg_dict[line_[0]] = line_[1]
    batch_path = "{}{}".format(arg_dict["visual_genome"], "batches")
    path_to_vocab_json = arg_dict["vocab"]
    logs_dir = arg_dict["logs"]
    batch_size = 32
    wgan = WGAN(batch_path, path_to_vocab_json, batch_size=batch_size)
    wgan.create_network()
    wgan.initialize_network(logs_dir)
    wgan.train_model(25)
