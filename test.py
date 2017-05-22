
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os, sys, inspect
import time
import json
#import utils
from tqdm import tqdm
import Image
import json
from scipy.ndimage import filters
from scipy.misc import imresize


from generator import Generator
from discriminator import Discriminator
from WGAN import WGAN

def computeImageMean(image_dir):
    if os.path.exists("preprocessing/image_mean.txt"):
        means = open("preprocessing/image_mean.txt", "r").read().strip().split()
        r_mean = float(means[0])
        g_mean = float(means[1])
        b_mean = float(means[2])
        return r_mean, g_mean, b_mean
    image_dir += "/" if image_dir[-1] != "/" else ""
    num_images = 0
    r_total = g_total = b_total = 0.0
    r_mean = g_mean = b_mean = 0.0
    for image_file in os.listdir(image_dir):
        #img = cv2.imread(image_dir + image_file).astype(np.float32)
        try:
            img = Image.open("{}{}".format(image_dir, image_file))
            img.load()
            img = np.array(img)
        except:
            #print "Error loading image"
            continue
        img = imresize(img, (224, 224, 3))
        img = np.array(img, dtype=np.float32)
        #Ignore grayscale images
        if img.shape != (224, 224, 3):
            continue
        r_total += np.sum(img[:,:,0]) / (img[:,:,0].shape[0] * img[:,:,0].shape[1])
        g_total += np.sum(img[:,:,1]) / (img[:,:,1].shape[0] * img[:,:,1].shape[1])
        b_total += np.sum(img[:,:,2]) / (img[:,:,2].shape[0] * img[:,:,2].shape[1])
        num_images += 1

    r_mean = r_total / float(num_images)
    g_mean = g_total / float(num_images)
    b_mean = b_total / float(num_images)

    with open("image_mean.txt", "w") as f:
        f.write("{} {} {}".format(r_mean, g_mean, b_mean))

    return r_mean, g_mean, b_mean

def smoothAndNormalizeImg(im, r_mean, g_mean, b_mean):
    im[:,:,0] -= r_mean
    im[:,:,1] -= g_mean
    im[:,:,2] -= b_mean
    im = filters.gaussian_filter(im, 2, mode='nearest')
    return im

def extractImageFeatures(path_to_image, tf_graph):
    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(tf_graph, input_map={"images" : im_placeholder})
    graph = tf.get_default_graph()
    im = Image.open(path_to_image)
    im.load()
    im = imresize(im, (224, 224, 3))
    #Imresize casts back to uint8 for some reason
    im = np.array(im, dtype=np.float64)
    r_mean, g_mean, b_mean = computeImageMean(image_dir)
    images = [smoothAndNormalizeImg(im, r_mean, g_mean, b_mean)]

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = { im_placeholder:np.array(images, dtype=np.float32) }
        feat_tensor = graph.get_tensor_by_name("import/conv5_3/Relu:0")
        feats = sess.run(feat_tensor, feed_dict = feed_dict)
    image_feats = np.reshape(feats, (feats.shape[0], -1, feats.shape[3]))

    return image_feats

def readInTensorflowModel(vgg_tf_model):
    #Load VGG Feature Extractor
    with open(vgg_tf_model, mode="rb") as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    return graph_def

if __name__ == "__main__":
    batch_path = "/home/user/data/visual_genome/batches"
    image_dir = "/home/user/data/visual_genome/all_images"
    path_to_vocab_json = "./preprocessing/vocab.json"
    path_to_image = "/home/user/data/visual_genome/all_images/1000.jpg"
    vgg_tf_model = "/home/user/misc_github_projects/Scene-Graph-GAN/models/vgg/vgg16.tfmodel"
    logs_dir = "./logs/"
    batch_size = 32
    #Extract the features of the image at the path first
    tf_graph = readInTensorflowModel(vgg_tf_model)
    image_feats = extractImageFeatures(path_to_image, tf_graph)
    #Now run those features through the WGAN
    wgan = WGAN(batch_path, path_to_vocab_json, batch_size=batch_size)
    wgan.create_network()
    wgan.initialize_network(logs_dir)
    #wgan.train_model(1)
    wgan.generate(image_feats, num_triples=25)
