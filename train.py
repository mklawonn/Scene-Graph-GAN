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
    def __init__(self, checkpoints_dir, summaries_dir, path_to_ims_to_triples, path_to_vocab, path_to_word_embeddings,\
                path_to_image_means, path_to_image_stds, critic_iters, batch_size, lambda_,resume):

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

        self.embeddings = np.load(path_to_word_embeddings)

        self.reverse_vocab = {y:x for x,y in self.vocab.iteritems()}

        with tf.variable_scope("Generator") as scope:
            self.g = Generator(len(self.vocab))

        with tf.variable_scope("Discriminator") as scope:
            self.embedding_placeholder = tf.placeholder(tf.float32, self.embeddings.shape)
            W = tf.Variable(tf.constant(0.0, shape=self.embeddings.shape), trainable=True, name="W")
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.d = Discriminator(len(self.vocab), W)

        self._loadImageMeans()

    def _Generator(self, images, is_training=True):
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE) as scope:
            generated_triple = self.g.build_generator(images, is_training)
            return generated_triple

    def _Discriminator(self, triple_input, images, is_training=True):
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE) as scope:
            logits = self.d.build_discriminator(triple_input, images, is_training)
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

        d = d.apply(tf.contrib.data.map_and_batch(
                map_func=self._parseFunction, batch_size=tf.cast(self.batch_size_placeholder, dtype=tf.int64), num_parallel_batches=2))


        #Shouldn't be necessary anymore since I can specify
        #The number of critic iters to run
        if train:
            d = d.flat_map(
                    lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(self.CRITIC_ITERS + 1))

        d = d.prefetch(buffer_size=tf.cast(self.batch_size_placeholder*(self.CRITIC_ITERS + 1)*2, dtype=tf.int64))

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
        #TODO Testing operation calculations
        self.global_step = tf.train.get_or_create_global_step()

        self.training_placeholder = tf.placeholder(tf.bool, shape=[])

        #fake_inputs = self._Generator(self.next_images, self.training_placeholder)
        #disc_real = self._Discriminator(self.next_labels, self.next_images, self.training_placeholder)

        wgan_model = tf.contrib.gan.gan_model(
                        self._Generator,
                        self._Discriminator,
                        real_data = self.next_labels,
                        generator_inputs = self.next_images)

        improved_wgan_loss = tf.contrib.gan.gan_loss(
                        wgan_model,
                        generator_loss_fn = tf.contrib.gan.losses.wasserstein_generator_loss,
                        discriminator_loss_fn = tf.contrib.gan.losses.wasserstein_discriminator_loss,
                        gradient_penalty_weight = self.LAMBDA,
                        gradient_penalty_one_sided = True)

        gen_cost = improved_wgan_loss[0]
        disc_cost = improved_wgan_loss[1]

        self.generator_optimizer =  tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name="Generator_Adam") 
        self.discriminator_optimizer =  tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name="Discriminator_Adam") 

        train_variables = tf.trainable_variables()
        gen_params = [v for v in train_variables if v.name.startswith("Generator")]
        disc_params = [v for v in train_variables if v.name.startswith("Discriminator")]

        self.gen_train_op = self.generator_optimizer.minimize(gen_cost, var_list = gen_params)
        self.disc_train_op = self.discriminator_optimizer.minimize(disc_cost, var_list = disc_params)

        #self.gan_train_ops = tf.contrib.gan.gan_train_ops(
        #    wgan_model,
        #    improved_wgan_loss,
        #    self.generator_optimizer,
        #    self.discriminator_optimizer)

        #self.train_steps = tf.contrib.gan.GANTrainSteps(1, self.CRITIC_ITERS)

        #self.train_fn = tf.contrib.gan.get_sequential_train_steps(self.train_steps)
        ##self.train_hooks = tf.contrib.gan.get_sequential_train_hooks(train_steps=self.train_steps)

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
            train_handle, val_handle = self._loadDatasets(sess)
            self._constructOps()

            if self.resume:
                self._loadModel(sess)
            else:
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_placeholder : self.embeddings})
                del self.embeddings

            pbar = tqdm(range(self.max_iterations))
            pbar.set_postfix(mode="Train")

            for itr in pbar:
                for i in range(self.CRITIC_ITERS):
                    sess.run(self.disc_train_op, feed_dict = {self.feedable_handle : train_handle, self.training_placeholder : True})
                sess.run(self.gen_train_op, feed_dict = {self.feedable_handle : train_handle, self.training_placeholder : True})


if __name__ == "__main__":
    
    ##################################################
    ## Argparse stuff                                #
    ##################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_dir", help="Where to save the tf checkpoints", default="./checkpoints")
    parser.add_argument("--summaries_dir", help="Where to write the logs", default="./logs")
    parser.add_argument("--path_to_ims_to_triples", help="Path to the map from image files to triples", default="./dataset_creation/ims_to_triples.json")
    parser.add_argument("--path_to_vocab", help="Path to the vocabulary", default = "./dataset_creation/vocab.json")
    parser.add_argument("--path_to_word_embeddings", help="Path to the word embeddings", default = "./dataset_creation/word_embeddings.npy")
    parser.add_argument("--path_to_image_means", help="Path to the image means", default="./dataset_creation/image_means.txt")
    parser.add_argument("--path_to_image_stds", help="Path to the image stds", default="./dataset_creation/image_stds.txt")

    parser.add_argument("--batch_size", default=64, help="Batch size defaults", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of critic iterations per generator iteration", type=int)
    parser.add_argument("--lambda", default=10, help="WGAN Lipschitz Penalty", type=int)
    parser.add_argument("--resume", default=False, help="Specifies whether or not to resume from the last checkpoint", type=bool)

    parser.add_argument("--GPU", default="0", help="Which GPU to use")

    args = parser.parse_args()
    params = vars(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params["GPU"])

    gan = SceneGraphGAN(params["checkpoints_dir"], params["summaries_dir"], params["path_to_ims_to_triples"], params["path_to_vocab"], params["path_to_word_embeddings"],\
                     params["path_to_image_means"], params["path_to_image_stds"], params["batch_size"], params["critic_iters"], params["lambda"], params["resume"])
    gan.train()
