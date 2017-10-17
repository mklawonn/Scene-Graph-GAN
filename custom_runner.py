import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np

import threading

from Queue import Queue

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, image_feat_dim, vocab_size, seq_len, batch_size, batch_path, dataset_relations_only, validation=False):

        self.image_feat_dim = image_feat_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.batch_path = batch_path
        queue_capacity = batch_size*10

        self.im_feats_placeholder = tf.placeholder(tf.float32, shape=[image_feat_dim[0], image_feat_dim[1]])
        self.triples_placeholder = tf.placeholder(tf.float32, shape=[seq_len, vocab_size])

        self.dataset_relations_only = dataset_relations_only
        self.validation = validation
        self.num_validation_images = 0

        min_after_dequeue = 0
        shapes = [[image_feat_dim[0], image_feat_dim[1]], [seq_len, vocab_size]]

        if validation:
            self.queue = tf.FIFOQueue(queue_capacity, dtypes=[tf.float32, tf.float32], shapes=shapes)
            self.gt_rels = Queue()
            self.gt_atts = Queue()
        else:
            self.queue = tf.RandomShuffleQueue(queue_capacity, min_after_dequeue, dtypes=[tf.float32, tf.float32], shapes=shapes)

        self.queue_size_op = self.queue.size()
        self.enqueue_op = self.queue.enqueue([self.im_feats_placeholder, self.triples_placeholder])

    def generateBigArr(self):
        train_path = os.path.join(self.batch_path, "train")
        filenames = [os.path.join(train_path, i) for i in os.listdir(train_path)]
        big_arr_list = []
        for f in range(len(filenames)):
            npz = np.load(filenames[f])
            big_arr_list.append(npz['arr_0'])
        return np.concatenate(big_arr_list, axis=0)

    def generateBigValArr(self):
        eval_path = os.path.join(self.batch_path, "eval")
        #filenames = [os.path.join(eval_path, i) for i in os.listdir(eval_path) if i[-4:] == ".npz"]
        #big_arr_list = []
        #for f in range(len(filenames)):
        #    npz = np.load(filenames[f])
        #    big_arr_list.append(npz['arr_0'])
        #return np.concatenate(big_arr_list, axis=0)
        npz = np.load(os.path.join(eval_path, "batch_0.npz"))
        big_arr = npz['arr_0']
        return big_arr
        
    def oneHot(self, trips):
        one_hot = np.zeros((trips.shape[0], self.seq_len, self.vocab_size), dtype=np.float32)
        for i in range(trips.shape[0]):
            for j in range(self.seq_len):
                one_hot[i, j, trips[i, j]] = 1.0
        return one_hot

    def dataGenerator(self):
        big_arr = self.generateBigArr()
        for i in range(0, big_arr.shape[0], 3):
            atts = big_arr[i+1]
            rels = big_arr[i+2]
            #Yield one_hot encoded attributes
            trips = self.oneHot(atts)
            im_feats = np.tile(np.expand_dims(big_arr[i], axis=0), (max(atts.shape[0], rels.shape[0]), 1, 1))
            for j in range(trips.shape[0]):
                yield im_feats[j], trips[j]
            #Yield one_hot encoded relations
            trips = self.oneHot(rels)
            for j in range(trips.shape[0]):
                yield im_feats[j], trips[j]

    def relationsOnlyDataGenerator(self):
        big_arr = self.generateBigArr()
        for i in range(0, big_arr.shape[0], 2):
            rels = big_arr[i+1]
            trips = self.oneHot(rels)
            im_feats = np.tile(np.expand_dims(big_arr[i], axis=0), (rels.shape[0], 1, 1))
            for j in range(trips.shape[0]):
                yield im_feats[j], trips[j]

    def relationsOnlyValidationGenerator(self):
        big_arr = self.generateBigValArr()
        self.num_validation_images = big_arr.shape[0] / 2
        for i in range(0, big_arr.shape[0], 2):
            gt_rels = big_arr[i+1]
            #Create some dummy relations in order to generate batch_size per image relations
            #This way each dequeue op will yield exactly batch_size copies of the same image feats
            #It will be the same image too, since we're using a fifo queue
            rels = np.ones((self.batch_size, 3), dtype=np.int64)
            trips = self.oneHot(rels)
            im_feats = np.tile(np.expand_dims(big_arr[i], axis=0), (rels.shape[0], 1, 1))
            #Queue up this batch of relations to compare
            self.gt_rels.put(gt_rels)
            for j in range(trips.shape[0]):
                yield im_feats[j], trips[j]

    def validationGenerator(self):
        print "Queue runner loading big_arr"
        big_arr = self.generateBigValArr()
        print "Big_arr loaded, {} elements".format(big_arr.shape[0])
        self.num_validation_images = big_arr.shape[0] / 3
        for i in range(0, big_arr.shape[0], 3):
            gt_rels = big_arr[i+1]
            gt_atts = big_arr[i+2]
            rels = np.ones((self.batch_size*2, 3), dtype=np.int64)
            trips = self.oneHot(rels)
            im_feats = np.tile(np.expand_dims(big_arr[i], axis=0), (self.batch_size*2, 1, 1))
            self.gt_rels.put(gt_rels)
            self.gt_atts.put(gt_atts)
            for j in range(trips.shape[0]):
                yield im_feats[j], trips[j]

    
    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        ims, triples = self.queue.dequeue_many(self.batch_size, name="DequeueOp")
        return ims, triples

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        #for i in range(self.num_epochs):
        while True:
            #TODO Support multiple enqueue threads?
            print "Spinning up the queue"
            if self.validation:
                if self.dataset_relations_only:
                    generator = self.relationsOnlyValidationGenerator()
                else:
                    print "The generator is for both attributes and relations"
                    generator = self.validationGenerator()
            else:
                if self.dataset_relations_only:
                    generator = self.relationsOnlyDataGenerator()
                else:
                    generator = self.dataGenerator()
            for im_batch, triples_batch in generator:
                feed_dict = {self.im_feats_placeholder : im_batch,\
                             self.triples_placeholder : triples_batch}
                sess.run(self.enqueue_op, feed_dict = feed_dict)


    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
