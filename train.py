import os, sys
sys.path.append(os.getcwd())

import time
import json
import random
import argparse
import threading
import architectures

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from subprocess import call
from custom_runner import CustomRunner

class SceneGraphWGAN(object):
    def __init__(self, batch_path, path_to_vocab_json, generator, discriminator, logs_dir, samples_dir,\
                 BATCH_SIZE=64, CRITIC_ITERS=10, LAMBDA=10, max_iterations=50000, convergence_threshold=5e-5,\
                 im_and_lang=False, resume=False, dataset_relations_only = False):
        #TODO Assert that im_and_lang isn't true if a language only discriminator/generator has been chosen
        self.batch_path = batch_path
        self.batch_path += "/" if self.batch_path[-1] != "/" else ""
        self.path_to_vocab_json = path_to_vocab_json
        self.path_to_vocab_json += "/" if self.path_to_vocab_json != "/" else ""
        self.discriminator = discriminator
        self.generator = generator
        if im_and_lang:
            self.configuration = "{}_gen_{}_disc_with_lang".format(generator, discriminator)
        else:
            self.configuration = "{}_gen_{}_disc".format(generator, discriminator)
        self.logs_dir = os.path.join(logs_dir, self.configuration)
        self.checkpoints_dir = os.path.join(self.logs_dir, "checkpoints/")
        self.summaries_dir = os.path.join(self.logs_dir, "summaries/")
        self.samples_dir = os.path.join(samples_dir, self.configuration)
        #self.num_epochs = num_epochs
        self.max_iterations = max_iterations
        self.queue_capacity = 5000
        self.queue_var_name = "queue_var"

        self.im_and_lang = im_and_lang
        self.resume = resume
        self.dataset_relations_only = dataset_relations_only

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        else:
            if not self.resume:
                print "WARNING: Checkpoints directory already exists for {} configuration and the resume flag was not specified. Files will be overwritten as necessary (not deleted though)".format(self.configuration)

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
        #self.decoder = {y[0]:x for x, y in self.vocab.iteritems()}
        self.decoder = {y:x for x, y in self.vocab.iteritems()}
        self.seq_len = 3

        #Image feature dimensionality
        self.image_feat_dim = [196, 512]
        self.word_embedding_size = 300
        #self.image_feat_dim = 4096

        #Hyperparameters
        self.BATCH_SIZE = int(BATCH_SIZE)
        self.LAMBDA = float(LAMBDA)
        self.CRITIC_ITERS = int(CRITIC_ITERS)
        self.DIM = 512
        self.convergence_threshold = convergence_threshold


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

        if self.im_and_lang:
            from architectures.cnn_language_discriminator import Discriminator as language_dim
            with tf.variable_scope("LanguageDiscriminator") as scope:
                self.language_d = language_dim(self.vocab_size, batch_size = self.BATCH_SIZE)

    def Generator(self, image_feats, batch_size, attribute_or_relation, prev_outputs=None):
        print "Building Generator"
        with tf.variable_scope("Generator", reuse=True) as scope:
            generated_words = self.g.build_generator(image_feats, batch_size, attribute_or_relation)
            return generated_words

    def Discriminator(self, triple_input, image_feats, batch_size, attribute_or_relation):
        print "Building Discriminator"
        with tf.variable_scope("Discriminator", reuse=True) as scope:
            logits = self.d.build_discriminator(image_feats, triple_input, batch_size, attribute_or_relation)
            if self.im_and_lang:
                lang_logits = self.language_d.build_discriminator(image_feats, triple_input, batch_size, attribute_or_relation)
                return tf.add(logits, lang_logits)
            else:
                return logits

    def constructOps(self):
        #Pin data ops to the cpu
        with tf.device("/cpu:0"):
            self.custom_runner = CustomRunner(self.image_feat_dim, self.vocab_size, self.seq_len, self.BATCH_SIZE, self.batch_path, self.dataset_relations_only)
            #self.inputs = self.custom_runner.get_inputs()
            ims, triples, flags = self.custom_runner.get_inputs()

        with tf.device("/gpu:0"):
            self.constant_ims = tf.get_variable("{}_ims".format(self.queue_var_name), initializer=ims, trainable=False)
            self.constant_triples = tf.get_variable("{}_triples".format(self.queue_var_name), initializer=triples, trainable=False)
            self.constant_flags = tf.get_variable("{}_flags".format(self.queue_var_name), initializer=flags, trainable=False)
            self.disc_step = tf.get_variable("{}_disc_step".format(self.queue_var_name), shape=[], initializer=tf.constant_initializer(0), trainable=False)

        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        #self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)

        self.global_step = tf.get_variable("global_step", shape=[], initializer=tf.constant_initializer(0), trainable=False)

        #disc_optimizer_decay = tf.train.exponential_decay(1e-2, self.disc_step, 10, .95, staircase=True)
        gen_optimizer_decay = tf.train.exponential_decay(1e-3, self.global_step, 10000, .9, staircase=True)
        #self.disc_optimizer = tf.train.GradientDescentOptimizer(disc_optimizer_decay)
        self.gen_optimizer = tf.train.GradientDescentOptimizer(gen_optimizer_decay)

        self.fake_inputs = self.Generator(self.constant_ims, self.BATCH_SIZE, self.constant_flags)

        disc_real = self.Discriminator(self.constant_triples, self.constant_ims, self.BATCH_SIZE, self.constant_flags)
        disc_fake = self.Discriminator(self.fake_inputs, self.constant_ims, self.BATCH_SIZE, self.constant_flags)

        disc_cost = tf.reduce_mean(disc_fake, axis=1) - tf.reduce_mean(disc_real, axis=1)
        disc_cost = tf.reduce_mean(disc_cost)

        LAMBDA = tf.constant(self.LAMBDA)
        
        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.BATCH_SIZE,1,1], 
            minval=0.,
            maxval=1.
        )
        differences = tf.subtract(self.fake_inputs, self.constant_triples)
        interpolates = tf.add(self.constant_triples, tf.multiply(alpha, differences))
        gradients = tf.gradients(self.Discriminator(interpolates, self.constant_ims, self.BATCH_SIZE, self.constant_flags), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.disc_cost = tf.add(disc_cost, tf.multiply(LAMBDA, gradient_penalty))

        train_variables = tf.trainable_variables()
        gen_params = [v for v in train_variables if v.name.startswith("Generator")]
        disc_params = [v for v in train_variables if v.name.startswith("Discriminator") or v.name.startswith("LanguageDiscriminator")]

        gen_cost = -tf.reduce_mean(disc_fake, axis=1)
        self.gen_cost = tf.reduce_mean(gen_cost)

        self.gen_train_op = self.gen_optimizer.minimize(self.gen_cost, var_list=gen_params, global_step=self.global_step)
        self.disc_train_op = self.disc_optimizer.minimize(self.disc_cost, var_list=disc_params, gate_gradients=self.disc_optimizer.GATE_GRAPH, global_step=self.disc_step)

        self.saver = tf.train.Saver()
                    
        
    def oneTrainingIteration(self, sess):
        #This tf_variables_initializer op runs the dequeue operation just once for all following operations
        #See https://stackoverflow.com/questions/43970221/how-to-read-the-top-of-a-queue-multiple-times-before-dequeueing-in-tensorflow
        queue_vars = [v for v in tf.global_variables() if self.queue_var_name in v.name]
        queue_init_op = tf.variables_initializer(queue_vars)
        sess.run(queue_init_op)

        old_disc_cost = -0.1
        diff = 10*self.convergence_threshold

        itr = 0
        while True:
            disc_cost, _= sess.run([self.disc_cost, self.disc_train_op])
            diff = np.abs(disc_cost - old_disc_cost)
            if (diff < self.convergence_threshold) and (disc_cost < 0):
            #if (diff < self.convergence_threshold):
                break
            old_disc_cost = disc_cost
            itr += 1
        #print itr

        gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op])
        return gen_cost, old_disc_cost


    def init(self):
        variables = [v for v in tf.global_variables() if self.queue_var_name not in v.name]
        return tf.variables_initializer(variables) 

    def loadModel(self, sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(self.checkpoints_dir, "model.ckpt")))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Problem loading model, exiting"
            sys.exit(1)

    def saveModel(self, sess, itr):
        self.saver.save(sess, os.path.join(self.checkpoints_dir, "model.ckpt"), global_step=self.global_step)
    
    def Train(self):
        self.constructOps()
        init_op = self.init()
        #Need to initialize in a non-naive way because some variables rely on the 
        #queue being populated, but you can't populate the queue until you initialize
        #the other variables. See init function for implementation.
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
            writer = tf.summary.FileWriter(self.summaries_dir, sess.graph)
            print "Initializing Variables"
            if self.resume:
                print "Resuming training, loading model"
                self.loadModel(sess)
                print "Model loaded, global step is {}".format(sess.run(self.global_step))
            else:
                sess.run(init_op)
            print "Done"
            print "Starting Queue"
            tf.train.start_queue_runners(sess=sess)
            self.custom_runner.start_threads(sess)
            print "Done"
            print "Training WGAN for {} iterations. To monitor training via TensorBoard, run python -m tensorflow.tensorboard --logdir {}"\
            .format(self.max_iterations, self.summaries_dir)
            pbar = tqdm(range(self.max_iterations))
            for itr in pbar:
                gen_cost, disc_cost = self.oneTrainingIteration(sess)
                pbar.set_description("Gen Cost: {} Disc Cost: {}".format(gen_cost, disc_cost))
                if itr % 1000 == 0:
                    self.saveModel(sess, itr)
            print "Done"
            self.saveModel(sess, itr)


if __name__ == "__main__":
    #Argparse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--vg_batches", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/saved_data/vocab.json", help="Path to the vocabulary")

    parser.add_argument("--batch_size", default=128, help="Batch size defaults to 128", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of iterations to train the critic", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--epochs", default=30, help="Number of epochs defaults to 30", type=int)
    parser.add_argument("--print_interval", default=500, help="The model will be saved and samples will be generated every <print_interval> iterations", type=int)
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")
    parser.add_argument("--lambda", default=10, help="Lambda term which regularizes to be close to one lipschitz", type=int)
    parser.add_argument("--use_language", default=False, help="Determines whether the generator update is also based on a discriminator trained on language only", type=bool)
    parser.add_argument("--resume", default=False, help="Resume training from the last checkpoint for this configuration", type=bool)
    parser.add_argument("--dataset_relations_only", default=False, help="When true, indicates that the data only contains relations, and will affect how data is read", type=bool)

    parser.add_argument("--GPU", default="0", help="Which GPU to use")

    args = parser.parse_args()
    params = vars(args)

    verbosity_dict = {"DEBUG" : 0, "INFO" : 1, "WARN" : 2, "ERROR" : 3}

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(verbosity_dict[params["tf_verbosity"]])
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = params["GPU"]

    #Call this just in case the graph is persisting due to TF closing suddenly
    tf.reset_default_graph()

    #Begin training
    wgan = SceneGraphWGAN(params["vg_batches"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"], resume=params["resume"],
           dataset_relations_only=params["dataset_relations_only"])
    #wgan.Train(params["epochs"], params["print_interval"])
    wgan.Train()
