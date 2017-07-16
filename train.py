import os, sys
sys.path.append(os.getcwd())

import time
import json
import random
import argparse
import threading

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from subprocess import call

class SceneGraphWGAN(object):
    def __init__(self, batch_path, path_to_vocab_json, generator, discriminator, logs_dir, samples_dir, BATCH_SIZE=64, CRITIC_ITERS=10, LAMBDA=10, num_epochs=1):
        self.batch_path = batch_path
        self.batch_path += "/" if self.batch_path[-1] != "/" else ""
        self.path_to_vocab_json = path_to_vocab_json
        self.path_to_vocab_json += "/" if self.path_to_vocab_json != "/" else ""
        self.configuration = "{}_gen_{}_disc_{}_critic".format(generator, discriminator, CRITIC_ITERS)
        self.logs_dir = os.path.join(logs_dir, self.configuration)
        self.checkpoints_dir = os.path.join(self.logs_dir, "checkpoints/")
        self.summaries_dir = os.path.join(self.logs_dir, "summaries/")
        self.samples_dir = os.path.join(samples_dir, self.configuration)
        self.num_epochs = num_epochs
        self.queue_capacity = 10000

        #For use with the data generator
        self.attributes_flag = 0.0
        self.relations_flag = 1.0

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        else:
            print "WARNING: Checkpoints directory already exists for {} configuration. Files will be overwritten.".format(self.configuration)

        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
        else:
            print "WARNING: Summaries directory already exists for {} configuration. Old files will be deleted.".format(self.configuration)

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)
        else:
            print "WARNING: Samples directory already exists for {} configuration. Old files will be deleted".format(self.configuration)

        for f in os.listdir(self.summaries_dir):
            call(["rm", os.path.join(self.summaries_dir, f)])

        for f in os.listdir(self.samples_dir):
            call(["rm", "-rf", os.path.join(self.samples_dir, f)])


        #Calculating vocabulary and sequence lengths
        with open(path_to_vocab_json, "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.decoder = {y[0]:x for x, y in self.vocab.iteritems()}
        self.seq_len = 3

        #Image feature dimensionality
        self.image_feat_dim = [196, 512]
        self.word_embedding_size = 300
        #self.image_feat_dim = 4096

        #Hyperparameters
        self.BATCH_SIZE = BATCH_SIZE
        self.LAMBDA = LAMBDA
        self.CRITIC_ITERS = CRITIC_ITERS
        self.DIM = 512
        self.ITERS = 100000
        self.convergence_threshold = 1e-5


        #Import the correct discriminator according to the keyword argument
        if discriminator == "language_only":
            from architectures.language_only_discriminator import Discriminator
        elif discriminator == "conv_lang":
            from architectures.cnn_language_discriminator import Discriminator
        elif discriminator == "conv1D":
            from architectures.conv1D_discriminator import Discriminator
        else:
            from architectures.discriminator_with_attention import Discriminator

        if generator == "language_only":
            from architectures.language_only_generator import Generator
        elif discriminator == "conv_lang":
            from architectures.cnn_language_generator import Generator
        elif generator == "conv1D":
            from architectures.conv1D_generator import Generator
        else:
            from architectures.generator_with_attention import Generator

        #Initialize all the generator and discriminator variables
        with tf.variable_scope("Generator") as scope:
            self.g = Generator(self.vocab_size, batch_size = self.BATCH_SIZE)

        with tf.variable_scope("Discriminator") as scope:
            self.d = Discriminator(self.vocab_size, batch_size = self.BATCH_SIZE)

    def Generator(self, image_feats, batch_size, attribute_or_relation, prev_outputs=None):
        print "Building Generator"
        with tf.variable_scope("Generator", reuse=True) as scope:
            generated_words = self.g.build_generator(image_feats, batch_size, attribute_or_relation)
            return generated_words

    def Discriminator(self, triple_input, image_feats, batch_size, attribute_or_relation):
        print "Building Discriminator"
        with tf.variable_scope("Discriminator", reuse=True) as scope:
            logits = self.d.build_discriminator(image_feats, triple_input, batch_size, attribute_or_relation)
            return logits


    def generateBigArr(self):
        train_path = os.path.join(self.batch_path, "train")
        filenames = [os.path.join(train_path, i) for i in os.listdir(train_path)]
        big_arr_1 = np.load(filenames[0])['arr_0']
        big_arr_list = []
        for f in range(1, len(filenames)):
            npz = np.load(filenames[f])
            big_arr_list.append(npz['arr_0'])
        return np.append(big_arr_1, big_arr_list)

    def oneHot(self, trips):
        one_hot = np.zeros((trips.shape[0], self.seq_len, self.vocab_size), dtype=np.float32)
        for i in range(trips.shape[0]):
            for j in range(self.seq_len):
                one_hot[i, j, trips[i, j]] = 1.0
        return one_hot

    def dataGenerator(self):
        big_arr = self.generateBigArr()
        for i in range(0, big_arr.shape[0], 3):
            #Yield one_hot encoded attributes
            trips = self.oneHot(big_arr[i+1])
            im_feats = np.tile(np.expand_dims(big_arr[i], axis=0), (trips.shape[0], 1, 1))
            flag = np.tile(self.attributes_flag, trips.shape[0])
            for i in range(trips.shape[0]):
                yield im_feats[i], trips[i], flag[i]
            #Yield one_hot encoded relations
            trips = self.oneHot(big_arr[i+2])
            flag = np.tile(self.relations_flag, trips.shape[0])
            for i in range(trips.shape[0]):
                yield im_feats[i], trips[i], flag[i]

    def constructOps(self):

        #Pin data ops to the cpu
        with tf.device("/cpu:0"):
            self.im_feats_placeholder = tf.placeholder(tf.float32, shape=[self.image_feat_dim[0], self.image_feat_dim[1]])
            self.triples_placeholder = tf.placeholder(tf.float32, shape=[self.seq_len, self.vocab_size])
            self.flag_placeholder = tf.placeholder(tf.float32, shape=[])

            min_after_dequeue = 0
            shapes = [[self.image_feat_dim[0], self.image_feat_dim[1]], [self.seq_len, self.vocab_size], []]
            self.queue = tf.RandomShuffleQueue(self.queue_capacity, min_after_dequeue, dtypes=[tf.float32, tf.float32, tf.float32], shapes=shapes)
            self.queue_size_op = self.queue.size()
            self.enqueue_op = self.queue.enqueue([self.im_feats_placeholder, self.triples_placeholder, self.flag_placeholder])

        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)

        self.train_while_loop = tf.while_loop(self.mainTrainCond, self.mainTrain, [tf.constant(0.0), tf.constant(0.0)])

    def enqueue(self, sess):
        #Enqueue data num_epochs times
        for i in range(self.num_epochs):
            #TODO Support multiple enqueue threads?
            generator = self.dataGenerator()
            for im_batch, triples_batch, flag_batch in generator:
                feed_dict = {self.im_feats_placeholder : im_batch,\
                             self.triples_placeholder : triples_batch,\
                             self.flag_placeholder : flag_batch}
                sess.run(self.enqueue_op, feed_dict = feed_dict)
    
    #while(condition(tensors)) { tensors = body(tensors); }
    def mainTrain(self, gen_cost, disc_cost):
        #Define enqueue and dequeue ops
        old_disc_cost = tf.constant(-0.1)
        diff = tf.constant(10*self.convergence_threshold)

        #ims, triples, flags = tf.train.batch(self.dequeue_op, batch_size=self.BATCH_SIZE, capacity=500, allow_smaller_final_batch=False, name="TheBatchOp")
        with tf.device("/cpu:0"):
            ims, triples, flags = self.queue.dequeue_many(self.BATCH_SIZE, name="TheBatchOp")

        fake_inputs = self.Generator(ims, self.BATCH_SIZE, flags)

        disc_real = self.Discriminator(triples, ims, self.BATCH_SIZE, flags)
        disc_fake = self.Discriminator(fake_inputs, ims, self.BATCH_SIZE, flags)

        def trainDiscToConvergence(old_disc_cost, diff):
            disc_cost = tf.reduce_mean(disc_fake, axis=1) - tf.reduce_mean(disc_real, axis=1)
            disc_cost = tf.reduce_mean(disc_cost)

            # WGAN lipschitz-penalty
            alpha = tf.random_uniform(
                shape=[self.BATCH_SIZE,1,1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_inputs - triples
            interpolates = triples + (alpha*differences)
            gradients = tf.gradients(self.Discriminator(interpolates, ims, self.BATCH_SIZE, flags), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += self.LAMBDA*gradient_penalty

            disc_train_op = self.disc_optimizer.minimize(disc_cost)
            diff = tf.abs(tf.subtract(disc_cost, old_disc_cost))
            with tf.control_dependencies([disc_train_op]):
                return disc_cost, diff

        def discConvergence(old_disc_cost, diff):
            return tf.less(diff, self.convergence_threshold)

        disc_while_loop = tf.while_loop(discConvergence, trainDiscToConvergence, [old_disc_cost, diff])

        # Compute the loss and gradient update based on the current example.
        gen_cost = -tf.reduce_mean(disc_fake, axis=1)
        gen_cost = tf.reduce_mean(gen_cost)
        gen_train_op = self.gen_optimizer.minimize(gen_cost)

        with tf.control_dependencies([disc_while_loop[0], gen_train_op]):
            return gen_cost, disc_while_loop[0]

    def mainTrainCond(self, gen_cost, disc_cost):
        return False

    def Train(self):
        self.constructOps()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            enqueue_thread = threading.Thread(target=self.enqueue, args=[sess])
            enqueue_thread.daemon = True
            enqueue_thread.start()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while True:
                    if coord.should_stop():
                        break
                    out = sess.run(self.train_while_loop)
                    print sess.run(self.queue_size_op)
            except Exception, e:
                print Exception
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            print "Done"
            #TODO: Save model


if __name__ == "__main__":
    #Argparse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/vocab.json", help="Path to the vocabulary")

    parser.add_argument("--batch_size", default=64, help="Batch size defaults to 64", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of iterations to train the critic", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--epochs", default=30, help="Number of epochs defaults to 30", type=int)
    parser.add_argument("--resume", default=False, help="Resume training from the last checkpoint for this configuration", type=bool)
    parser.add_argument("--print_interval", default=500, help="The model will be saved and samples will be generated every <print_interval> iterations", type=int)
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")
    parser.add_argument("--lambda", default=10, help="Lambda term which regularizes to be close to one lipschitz", type=int)

    args = parser.parse_args()
    params = vars(args)

    batch_path = os.path.join(params["visual_genome"], "batches")

    verbosity_dict = {"DEBUG" : 0, "INFO" : 1, "WARN" : 2, "ERROR" : 3}

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(verbosity_dict[params["tf_verbosity"]])

    #Begin training
    wgan = SceneGraphWGAN(batch_path, params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"])
    #wgan.Train(params["epochs"], params["print_interval"])
    wgan.Train()
