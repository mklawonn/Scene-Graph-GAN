#Code based on https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW/blob/master/main.py


import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope

import os
import ipdb
import numpy as np
import pandas as pd
import cPickle
import scipy.misc
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar

from gan import GAN


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch_size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan")


FLAGS = flags.FLAGS

if __name__ == "__main__":
    #TODO Figure out the actual data directory
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedires(data_directory)
    
    model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    #Main training loop
    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            #TODO change this portion of the code to read in a random batch of 
            #pre-computed VGG features paired with assertions
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            #TODO Generate images here
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss {}".format(training_loss))

        #TODO Make sure to change the generation from images to text
        #Note this will be done in the gan.py file
        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)
"""Code from here on down is the training code that came with the generator"""

###### Parameters ######
n_epochs=1000
batch_size=80
dim_embed=256
dim_context=512
dim_hidden=256
context_shape=[196,512]
pretrained_model_path = './model/model-8'
#############################
###### Parameters #####
annotation_path = './data/annotations.pickle'
feat_path = './data/feats.npy'
model_path = './model/'
#############################


def train(pretrained_model_path=pretrained_model_path):
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    learning_rate=0.001
    vocab_size = len(wordtoix)
    feats = np.load(feat_path)
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            vocab_size=vocab_size,
            dim_embed=dim_embed,
            dim_context=dim_context,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1, #
            batch_size=batch_size,
            context_shape=context_shape,
            bias_init_vector=bias_init_vector)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(sess, pretrained_model_path)

    index = list(annotation_data.index)
    np.random.shuffle(index)
    annotation_data = annotation_data.ix[index]

    captions = annotation_data['caption'].values
    image_id = annotation_data['image_id'].values

    for epoch in range(n_epochs):
        for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            current_feats = feats[ image_id[start:end] ]
            current_feats = current_feats.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1,2)

            current_captions = captions[start:end]
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions) #

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})

            print "Current Cost: ", loss_value
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

def test(test_feat='./guitar_player.npy', model_path='./model/model-6', maxlen=20):
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    vocab_size = len(wordtoix)
    feat = np.load(test_feat).reshape(-1, context_shape[1], context_shape[0]).swapaxes(1,2)

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            vocab_size=vocab_size,
            dim_embed=dim_embed,
            dim_context=dim_context,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen,
            batch_size=batch_size,
            context_shape=context_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=maxlen)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index = sess.run(generated_words, feed_dict={context:feat})
    alpha_list_val = sess.run(alpha_list, feed_dict={context:feat})
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    alpha_list_val = alpha_list_val[:punctuation]
    return generated_words, alpha_list_val

#    generated_sentence = ' '.join(generated_words)
#    ipdb.set_trace()
