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

def buildVocab(path_to_data):
    if os.path.exists("vocab.json"):
        js = open("vocab.json").read()
        return json.loads(js)
    to_index_dict = {}
    to_index_dict["is"] = 0
    path_to_data += "/" if path_to_data[-1] != "/" else ""

    scene_graphs = vg.GetSceneGraphs(startIndex=0, endIndex=-1,
                   dataDir=path_to_data, imageDataDir="{}by_id/".format(path_to_data),
                   minRels=0, maxRels=100)
    for g in scene_graphs:
        for i in g.relationships:
            if str(i.subject).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            if str(i.predicate).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            if str(i.object).lower() not in to_index_dict.keys():
                to_index_dict[str(i.object).lower()] = len(to_index_dict.keys())
        for o in g.objects:
            if str(o).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            for attribute in o.attributes:
                if str(attribute).lower() not in to_index_dict.keys():
                    to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
    with open("vocab.json", "w") as f:
        json.dump(to_index_dict, f)
    
    return to_index_dict

def encodeTriple(vocab, triple):
    #This is a dense representation of the sequence 
    triple = triple.split()
    dense = [0]*len(triple)
    for i in xrange(len(triple)):
        dense[i] = vocab[triple[i]]
    return dense

def computeImageMean(image_dir):
    if os.path.exists("image_mean.txt"):
        means = open("image_mean.txt", "r").read().strip().split()
        r_mean = means[0]
        g_mean = means[1]
        b_mean = means[2]
        return r_mean, g_mean, b_mean
    image_dir += "/" if image_dir[-1] != "/" else ""
    num_images = 0
    r_total = g_total = b_total = 0.0
    r_mean = g_mean = b_mean = 0.0
    for image_file in os.listdir(image_dir):
        #img = cv2.imread(image_dir + image_file).astype(np.float32)
        try:
            img = np.array(Image.open("{}{}".format(image_dir, image_file)))
        except:
            continue
        img = imresize(img, (224, 224, 3))
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

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def readInTensorflowModel(vgg_tf_model):
    print "Loading feature extractor"
    #Load VGG Feature Extractor
    with open(vgg_tf_model, mode="rb") as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={"images" : im_placeholder})

    graph = tf.get_default_graph()
    return graph
 

def imageDataGenerator(path_to_data, image_files, graph, image_means, vocab, chunk_size = 2500):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    path_to_images = path_to_data + "all_images/"
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    r_mean = image_means[0]
    g_mean = image_means[1]
    b_mean = image_means[2]
    for group in grouper(image_files, chunk_size):
        ids = []
        attributes_relationships = []
        images = []
        for f in group:
            if f == None:
                break
            try:
                im = np.array(Image.open(f), dtype=np.float32)
            except:
                continue
            im = imresize(im, (224, 224, 3))
            #Skip grayscale images
            if img.shape != (224, 224, 3):
                continue
            ids.append(int(f[f.rfind("/")+1:-4]))
            graph = vg.GetSceneGraph(ids[-1], images=path_to_images, imageDataDir="{}by_id/".format(path_to_data))
            #First add the relationships
            attributes_relationships.append([encodeTriple("{} {} {}".format(str(i.subject), str(i.predicate).lower(), str(i.object))) \
                             for i in graph.relationships])
            for o in graph.objects:
                for attribute in o.attributes:
                    attributes_relationships[-1].append(encodeTriple("{} is {}".format(o, attribute)))
            im[:,:,0] -= r_mean
            im[:,:,1] -= g_mean
            im[:,:,2] -= b_mean
            im = filters.gaussian_filter(im, 2, mode='nearest')
            images.append(im)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            feed_dict = {im_placeholder:images}
            feat_tensor = graph.get_tensor_by_name("import/conv5_3/Relu:0")
            feats = sess.run(feat_tensor)
        feats = np.reshape(feats, (-1, feats.shape[2]))
        #feats = np.array(feats, dtype=np.float32)
        yield feats, ids, attributes_relations

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
    r_mean, g_mean, b_mean = computeImageMean(path_to_images)
    image_means = [r_mean, g_mean, b_mean]

    image_files = ["{}{}.jpg".format(path_to_images, im_path) \
            for im_path in sorted([int(i.replace(".jpg", "")) for i in os.listdir(path_to_images)])]

    graph = readInTensorflowModel(vgg_tf_model)
    vocab = buildVocab(path_to_data)

    im_generator = imageDataGenerator(path_to_data, image_files, graph, image_means, vocab)
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
