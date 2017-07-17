import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf

import threading

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, image_feat_dim, vocab_size, seq_len, batch_size, num_epochs):

        self.batch_size = batch_size
        queue_capacity = batch_size*4
        self.num_epochs = num_epochs


        self.im_feats_placeholder = tf.placeholder(tf.float32, shape=[image_feat_dim[0], image_feat_dim[1]])
        self.triples_placeholder = tf.placeholder(tf.float32, shape=[seq_len, vocab_size])
        self.flag_placeholder = tf.placeholder(tf.float32, shape=[])

        min_after_dequeue = 0
        shapes = [[image_feat_dim[0], image_feat_dim[1]], [seq_len, vocab_size], []]
        self.queue = tf.RandomShuffleQueue(queue_capacity, min_after_dequeue, dtypes=[tf.float32, tf.float32, tf.float32], shapes=shapes)
        self.queue_size_op = self.queue.size()
        self.enqueue_op = self.queue.enqueue([self.im_feats_placeholder, self.triples_placeholder, self.flag_placeholder])

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


    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        ims, triples, flags = self.queue.dequeue_many(self.batch_size, name="DequeueOp")
        return ims, triples, flags

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for i in range(self.num_epochs):
            #TODO Support multiple enqueue threads?
            generator = self.dataGenerator()
            for im_batch, triples_batch, flag_batch in generator:
                feed_dict = {self.im_feats_placeholder : im_batch,\
                             self.triples_placeholder : triples_batch,\
                             self.flag_placeholder : flag_batch}
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
