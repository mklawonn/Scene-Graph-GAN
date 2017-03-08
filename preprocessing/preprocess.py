import local as vg
import tensorflow as tf
import numpy as np
import os
import Image
import json
from scipy.ndimage import filters
from scipy.misc import imresize
from itertools import izip_longest
from tqdm import tqdm

def computeImageMean(image_dir):
    if os.path.exists("image_mean.txt"):
        means = open("image_mean.txt", "r").read().strip().split()
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

def buildVocab(path_to_data):
    if os.path.exists("./vocab.json"):
        with open("./vocab.json", "r") as f:
            vocab = json.load(f)
        return vocab
    vocab = {}
    vocab[u"is"] = 0
    with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}

    for i in tqdm(sg_dict):
        for o in sg_dict[i]["objects"]:
            if "synsets" in o:
                if len(o["synsets"]) > 0:
                    wordnet_synset = o["synsets"][0]
                    if not wordnet_synset in vocab:
                        vocab[wordnet_synset] = len(vocab)
                elif "names" in o:
                    name = o["names"][0]
                    if not name in vocab:
                        vocab[name] = len(vocab)
            elif "names" in o:
                name = o["names"][0]
                if not name in vocab:
                    vocab[name] = len(vocab)
            if "attributes" in o:
                for attribute in o["attributes"]:
                    if not attribute in vocab:
                        vocab[attribute] = len(vocab)
        for r in sg_dict[i]["relationships"]:
            if "synsets" in r:
                if len(r["synsets"]) > 0:
                    predicate = r["synsets"][0]
                else:
                    predicate = r["predicate"]
            else:
                predicate = r["predicate"]
            if not predicate in vocab:
                vocab[predicate] = len(vocab)
            #The subject and object should both already be added
            #via the objects loop above

    with open("./vocab.json", "w") as f:
        json.dump(vocab, f)

    return vocab

#For a single scene graph, return the attribute and relationship triples
def parseSceneGraph(sg):
    attributes = []
    relationships = []
    ob_id_dict = {}
    for o in sg["objects"]
        if "synsets" in o:
            if len(o["synsets"]) > 0:
                name = o["synsets"][0]
            elif "names" in o:
                name = o["names"][0]
        elif "names" in o:
            name = o["names"][0]
        ob_id_dict[o["object_id"]] = name
        if "attributes" in o:
            for attribute in o["attributes"]:
                attributes.append((name, u"is", attribute))
    for r in sg_dict[i]["relationships"]:
        if "synsets" in r:
            if len(r["synsets"]) > 0:
                predicate = r["synsets"][0]
            else:
                predicate = r["predicate"]
        else:
            predicate = r["predicate"]
        relationships.append((ob_id_dict[r["subject_id"]], predicate, ob_id_dict[r["object_id"]]))
    return attributes, relationships

def encodeTriple(vocab, subject, predicate, object):
    #This is a dense representation of the sequence 
    dense = [0,0,0]
    dense[0] = vocab[subject]
    dense[1] = vocab[predicate]
    dense[2] = vocab[object]
    return dense

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def readInTensorflowModel(vgg_tf_model):
    #Load VGG Feature Extractor
    with open(vgg_tf_model, mode="rb") as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={"images" : im_placeholder})

    graph = tf.get_default_graph()
    return graph

def smoothAndNormalizeImg(im, r_mean, g_mean, b_mean):
    im[:,:,0] -= r_mean
    im[:,:,1] -= g_mean
    im[:,:,2] -= b_mean
    im = filters.gaussian_filter(im, 2, mode='nearest')
    return im
 

def imageDataGenerator(path_to_data, image_files, tf_graph, image_means, vocab, chunk_size = 2500):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    r_mean = image_means[0]
    g_mean = image_means[1]
    b_mean = image_means[2]

    with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}

    for group in grouper(sg_dict, chunk_size):
        attributes_relationships = []
        images = []
        for im_id in tqdm(group):
            if im_id == None:
                break
            im = Image.open(image_files[im_id])
            im.load()
            im = imresize(im, (224, 224, 3))
            #Imresize casts back to uint8 for some reason
            im = np.array(im, dtype=np.float32)
            #Skip grayscale images
            if im.shape != (224, 224, 3):
                continue
            images.append(smoothAndNormalizeImg(im, r_mean, g_mean, b_mean))
            attributes, relations = parseSceneGraph(sg_dict[im_id])
            attributes_relationships.append((attributes, relations))
        im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            feed_dict = {im_placeholder:images}
            feat_tensor = tf_graph.get_tensor_by_name("import/conv5_3/Relu:0")
            feats = sess.run(feat_tensor)
        feats = np.reshape(feats, (-1, feats.shape[2]))
        #feats = np.array(feats, dtype=np.float32)
        yield feats, group, attributes_relations

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[int_64_feature(v) for v in values])

def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

#Inspired by https://github.com/tensorflow/models/blob/master/im2txt/im2txt/data/build_mscoco_data.py
def toTFRecord(path_to_data, vgg_tf_model):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    path_to_images = path_to_data + "all_images/"
    print "Computing Image Mean"
    r_mean, g_mean, b_mean = computeImageMean(path_to_images)
    image_means = [r_mean, g_mean, b_mean]
    print "Done"

    with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}

    image_files = ["{}{}.jpg".format(path_to_images, im_id) \
            for im_id in sorted(sg_dict)]

    print "Reading in tensorflow model"
    graph = readInTensorflowModel(vgg_tf_model)
    print "Done"
    print "Building vocabulary"
    vocab = buildVocab(path_to_data)
    print "Done"

    print "Creating image generator"
    im_generator = imageDataGenerator(path_to_data, image_files, graph, image_means, vocab)
    print "Done"
    count = 0
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    for image_feats, id_batch, att_rels_batch in im_generator:
        path_to_tf_records = "{}tf_records/batch_{}.tfrecords".format(path_to_data, count)
        count += 1
        writer = tf.python_io.TFRecordWriter(path_to_tf_records)
        for i in xrange(len(id_batch)):
            im_raw = image_feats[i].tostring()
            num_feats = image_feats[i].shape[0] 
            feat_dim = image_feats[i].shape[1]
            #TODO: Rather than truncating to 10 you could pad up to the max
            #And then during training ignore the padding triples
            #TODO: You could at least make this random
            #TODO Make sure this is still the way you should be reading this in
            first_ten = [bytearray(att_rels_batch[j]) for j in xrange(10)]
            #Construct the Example proto object
            context = tf.train.Features(feature={
                "image/image_id": _int64_feature(id_batch[i]),
                "image/data": _bytes_feature(im_raw),
            })
            feature_lists = tf.train.FeatureLists(feature_list={
                "image/triples": _bytes_feature_list(first_ten),
            })
            sequence_example = tf.train.SequenceExample(
                context=context, feature_lists=feature_lists)
            """example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'atts_and_rels': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=att_rels_batch[i])),
                        'image': tf.train.Feature(
                            bytes_list = tf.train.BytesList(value=im_raw)),
                        'im_id': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[id_batch[i]])),
                        'num_feats': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[num_feats])),
                        'feat_dim': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[feat_dim]))
            }))"""
            serialized = sequence_example.SerializeToString()
            writer.write(serialized)
        writer.close()
    return


if __name__ == "__main__":
    vgg_tf_model = "/home/user/misc_github_projects/Scene-Graph-GAN/models/vgg/vgg16.tfmodel"
    path_to_data = "/home/user/data/visual_genome/"
    toTFRecord(path_to_data, vgg_tf_model)
