import os, sys
sys.path.append(os.getcwd())

import json
import argparse

import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle
from subprocess import call
from tqdm import tqdm

from architectures.generator_with_attention import Generator
from architectures.discriminator_with_attention import Discriminator

class SceneGraphGAN(object):


    ############################################################
    ## All init methods
    ############################################################
    def __init__(self, checkpoints_dir, summaries_dir, path_to_ims_to_triples, path_to_vocab, path_to_image_means, path_to_image_stds,\
                critic_iters, batch_size, lambda_,resume):

        #Hyperparameters
        self.CRITIC_ITERS = critic_iters
        self.BATCH_SIZE = batch_size
        self.VAL_BATCH_SIZE = batch_size / 2
        self.TEST_BATCH_SIZE = batch_size
        self.LAMBDA = lambda_

        #Flags
        self.resume = resume

        #Create paths
        self.path_to_ims_to_triples = path_to_ims_to_triples
        self.path_to_vocab = path_to_vocab
        self.path_to_image_means = path_to_image_means
        self.path_to_image_stds = path_to_image_stds
        self.checkpoints_dir = checkpoints_dir
        self.summaries_dir = summaries_dir

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        for f in os.listdir(self.summaries_dir):
            call(["rm", os.path.join(self.summaries_dir, f)])
        for f in os.listdir(self.checkpoints_dir):
            call(["rm", os.path.join(self.checkpoints_dir, f)])

        with open(path_to_ims_to_triples, "r") as f:
            self.ims_to_triples = json.load(f)

        with open(path_to_vocab, "r") as f:
            self.vocab = json.load(f)

        self.reverse_vocab = {y:x for x,y in self.vocab.iteritems()}

        with tf.variable_scope("Generator") as scope:
            self.g = Generator(len(self.vocab))

        with tf.variable_scope("Discriminator") as scope:
            self.d = Discriminator(len(self.vocab))

        self._loadImageMeans()

    def _Generator(self, images, training):
        with tf.variable_scope("Generator", reuse=True) as scope:
            generated_triple = self.g.build_generator(images, training)
            return generated_triple

    def _Discriminator(self, triple_input, images):
        with tf.variable_scope("Discriminator", reuse=True) as scope:
            logits = self.d.build_discriminator(images, triple_input)
            return logits

    def _loadImageMeans(self):
        image_means = []
        image_stds = []
        with open(self.path_to_image_means, "r") as f:
            for line in f:
                image_means.append(float(line.strip()))
        with open(self.path_to_image_stds, "r") as f:
            for line in f:
                image_stds.append(float(line.strip()))
        #Create tf constants
        self.image_means = np.ones((221,221,3), dtype=np.float64)
        self.image_means = tf.constant(image_means*self.image_means, dtype=tf.float32)
        self.image_stds = np.ones((221,221,3), dtype=np.float64)
        self.image_stds = tf.constant(image_stds*self.image_stds, dtype=tf.float32)

    ############################################################
    ## All data loading ops
    ############################################################

    def _gatherFiles(self):
        train_files = []
        train_labels = []
        val_files = []
        val_labels = [] 
        test_files = []
        test_labels = []
        all_files = []
        all_labels = []

        for k,v in self.ims_to_triples.iteritems():
            for t in v:
                all_files.append(k)
                all_labels.append(t)

        all_files, all_labels = shuffle(all_files, all_labels)
        train_threshold = int(0.8*len(all_files))
        val_threshold = int(0.9*len(all_files))
        train_files = all_files[:train_threshold]
        train_labels = np.reshape(all_labels[:train_threshold], (-1, 3))
        val_files = all_files[train_threshold:val_threshold]
        val_labels = np.reshape(all_labels[train_threshold:val_threshold], (-1, 3))
        test_files = all_files[val_threshold:]
        test_labels = np.reshape(all_labels[val_threshold:], (-1, 3))

        self.max_iterations = 5*len(train_files)
        self.write_iterations = int(len(train_files) / 50)
        self.test_iterations = int(len(train_files) / 2)

        del all_files
        del all_labels

        assert len(train_files) > len(val_files)

        return train_files, train_labels, val_files, val_labels, test_files, test_labels

    def _parseFunction(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [221, 221])
        #image_standardized = tf.image.per_image_standardization(image_resized)
        image_standardized = (image_resized - self.image_means) / self.image_stds
        one_hot = tf.one_hot(label, len(self.vocab))
        return image_standardized, one_hot

    def _createSingleDataset(self, train=True, val=False, test=False):
        d = tf.data.Dataset.from_tensor_slices((self.files_placeholder, self.labels_placeholder))
        if train or val:
            d = d.repeat()
            d = d.shuffle(buffer_size=tf.cast(self.batch_size_placeholder*10, dtype=tf.int64))

        d = d.prefetch(buffer_size=tf.cast(self.batch_size_placeholder, dtype=tf.int64))
        d = d.apply(tf.contrib.data.map_and_batch(
                map_func=self._parseFunction, batch_size=tf.cast(self.batch_size_placeholder, dtype=tf.int64), num_parallel_batches=2))

        if train:
            d = d.flat_map(
                    lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.CRITIC_ITERS + 1))

        iterator = d.make_initializable_iterator()
        return d, iterator

    def _loadDatasets(self, sess):
        train_files, train_labels, val_files, val_labels, test_files, test_labels = self._gatherFiles()

        self.files_placeholder = tf.placeholder(tf.string, shape=[None])
        self.labels_placeholder = tf.placeholder(tf.int64, shape=[None, 3])
        self.batch_size_placeholder = tf.placeholder(tf.int32, shape=[])

        self.train_d, self.train_it = self._createSingleDataset(train=True)
        self.val_d, self.val_it = self._createSingleDataset(train=False, val=True)
        self.test_d, self.test_it = self._createSingleDataset(train=False, test=True)

        self._initDataset(self.train_it, sess, train_files, train_labels, self.BATCH_SIZE)
        self._initDataset(self.val_it, sess, val_files, val_labels, self.VAL_BATCH_SIZE)

        self._createFeedables(self.train_d.output_types, self.train_d.output_shapes)
        
        train_handle = sess.run(self.train_it.string_handle())
        val_handle = sess.run(self.val_it.string_handle())

        #TODO Make sure that the test dataset is
        #compatible with self.test() and the test
        #operations
        self.test_files = test_files
        self.test_labels = test_labels

        return train_handle, val_handle


    def _createFeedables(self, output_types, output_shapes):
        self.feedable_handle = tf.placeholder(tf.string, shape=[])

        self.feedable_iterator = tf.data.Iterator.from_string_handle(self.feedable_handle, output_types, output_shapes)
        self.next_images, self.next_labels = self.feedable_iterator.get_next()

    def _initDataset(self, iterator, sess, files, labels, batch_size):
        sess.run(iterator.initializer, feed_dict={self.files_placeholder : files, self.labels_placeholder : labels, self.batch_size_placeholder : batch_size})

    ############################################################
    ## Graph construction
    ############################################################
    def _constructOps(self):
        #Use self.next_images and self.next_labels
        #TODO Load Architectures
        #TODO WGAN Objective
        #TODO Test operation calculations
        self.training_placeholder = tf.placeholder(tf.bool, shape=[])

        fake_inputs = self.Generator(self.next_images, self.training_placeholder)
        sys.exit(1)
        #fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)
                                                                                                                                                                 
        self.fake_inputs = fake_inputs
                                                                                                                                                                 
        #TODO Uncomment this to test the loss functions
        #gen_cost = tf.contrib.gan.losses.wargs.wasserstein_generator_loss(
        #    disc_fake)

        #disc_cost = tf.contrib.gan.losses.wargs.wassertein_gradient_penalty(
        #        self.next_labels, self.fake_inputs, self.next_images,\
        #        self.discriminator_function, "Discriminator", one_sided=True
        #    )
        #                                                                                                                                                                 
        #train_variables = tf.trainable_variables()
        #gen_params = [v for v in train_variables if v.name.startswith("Generator")]
        #disc_params = [v for v in train_variables if v.name.startswith("Discriminator")]

        ##Optimizer
        #self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name="Generator_Adam").minimize(gen_cost, var_list=gen_params)
        #self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name="Discriminator_Adam").minimize(disc_cost, var_list=disc_params)

    ############################################################
    ## Saving and Testing
    ############################################################

    def _saveModel(self, sess, itr):
        self.saver.save(sess, os.path.join(self.checkpoints_dir, "model.ckpt"), global_step=self.global_step)

    def _loadModel(self, sess):
        self.saver.restore(sess, os.path.join(self.checkpoints_dir, "model.ckpt"))

    #TODO Implement testing
    def test(self, sess):
        self._initDataset(self.test_it, sess, self.test_files, self.test_labels, self.TEST_BATCH_SIZE)
        test_handle = sess.run(self.test_it.string_handle())

    ############################################################
    ## Training
    ############################################################

    def train(self):

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.summaries_dir, sess.graph)

            print "Loading Datasets"
            train_handle, val_handle = self._loadDatasets(sess)
            print "Constructing Graph"
            self._constructOps()

            if self.resume:
                self._loadModel(sess)
            else:
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init)
            
            print "Starting training"
            pbar = tqdm(range(self.max_iterations))
            pbar.set_postfix(mode="Train")
            for itr in pbar:
                for i in range(self.CRITIC_ITERS):
                    try:
                        sess.run(self.disc_train_op, feed_dict = {self.training_placeholder : True, self.feedable_handle : train_handle})
                    except:
                        break

                sess.run(self.gen_train_op, feed_dict = {self.training_placeholder : True, self.feedable_handle : train_handle})
                if itr % self.write_iterations == 0:
                    pbar.set_postfix(mode="Val")
                    #merged = sess.run(self.merged, feed_dict = {self.training_placeholder : False, self.feedable_handle : val_handle})
                    #train_writer.add_summary(merged, itr)
                    pbar.set_postfix(mode="Train")

                if itr % self.test_iterations == self.test_iterations - 1:
                    pbar.set_postfix(mode="Test")
                    self.test(sess)
                    pbar.set_postfix(mode="Train")


if __name__ == "__main__":
    
    ##################################################
    ## Argparse stuff                                #
    ##################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_dir", help="Where to save the tf checkpoints", default="./checkpoints")
    parser.add_argument("--summaries_dir", help="Where to write the logs", default="./logs")
    parser.add_argument("--path_to_ims_to_triples", help="Path to the map from image files to triples", default="./dataset_creation/ims_to_triples.json")
    parser.add_argument("--path_to_vocab", help="Path to the vocabulary", default = "./dataset_creation/vocab.json")
    parser.add_argument("--path_to_image_means", help="Path to the image means", default="./dataset_creation/image_means.txt")
    parser.add_argument("--path_to_image_stds", help="Path to the image stds", default="./dataset_creation/image_stds.txt")

    parser.add_argument("--batch_size", default=64, help="Batch size defaults", type=int)
    parser.add_argument("--critic_iters", default=12, help="Number of critic iterations per generator iteration", type=int)
    parser.add_argument("--lambda", default=10, help="WGAN Lipschitz Penalty", type=int)
    parser.add_argument("--resume", default=False, help="Specifies whether or not to resume from the last checkpoint", type=bool)

    parser.add_argument("--GPU", default="0", help="Which GPU to use")

    args = parser.parse_args()
    params = vars(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params["GPU"])

    gan = SceneGraphGAN(params["checkpoints_dir"], params["summaries_dir"], params["path_to_ims_to_triples"], params["path_to_vocab"],\
                     params["path_to_image_means"], params["path_to_image_stds"], params["batch_size"], params["critic_iters"], params["lambda"], params["resume"])
    gan.train()
