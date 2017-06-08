import os, sys
sys.path.append(os.getcwd())

import preprocessing.local as vg
import tensorflow as tf
import numpy as np
import os
import PIL.Image as Image
import json
from scipy.ndimage import filters
from scipy.misc import imresize
from itertools import izip_longest
from tqdm import tqdm

def computeImageMean(image_dir):
    if os.path.exists("./preprocessing/image_mean.txt"):
        means = open("./preprocessing/image_mean.txt", "r").read().strip().split()
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

    with open("./preprocessing/image_mean.txt", "w") as f:
        f.write("{} {} {}".format(r_mean, g_mean, b_mean))

    return r_mean, g_mean, b_mean

#TODO Must adapt to work on both conv and fc features
def normalizeFeatures(batch_dir, which_to_extract = "conv"):
    if which_to_extract != "conv":
        print "Feature extraction mode {} is not currently supported".format(which_to_extract)
        sys.exit(1)

    #Go through all batches once and calculate the min, max, and means
    print "Calculating feature statistics"
    mean = np.zeros((512))
    std_dev = np.zeros((512))
    file_count = 0.0
    for f in os.listdir(batch_dir):
        npz = np.load(os.path.join(batch_dir, f))
        big_arr = npz['arr_0']
        num_ims = 0.0
        temp_mean = np.zeros((512))
        temp_std_dev = np.zeros((512))
        for i in xrange(0, big_arr.shape[0], 2):
            #Im feats is 196 x 512 when conv is selected
            im_feats = big_arr[i]
            #Calculate the per channel mean
            temp_mean = temp_mean + np.mean(im_feats, axis=0)
            #Calculate the per channel std deviation
            temp_std_dev = temp_std_dev + np.std(im_feats, axis=0)
            num_ims += 1.0
        mean = mean + temp_mean
        std_dev = std_dev + temp_std_dev
        file_count += 1.0
            
    mean = mean / file_count
    std_dev = std_dev / file_count
    print "Done"

    print "Normalizing features"
    #Now center and normalize the data
    for f in os.listdir(batch_dir):
        npz = np.load(os.path.join(batch_dir, f))
        big_arr = npz['arr_0']
        for i in xrange(0, big_arr.shape[0], 2):
            big_arr[i] -= mean
            big_arr[i] *= (1./std_dev)
        np.savez(os.path.join(batch_dir, f), big_arr)
    print "Done"

#When the vocab is built, the first entry is the index and the second entry is
#a count indicating how many times that word appears in the vocabulary
def buildVocab(sg_dict):
    if os.path.exists("./preprocessing/vocab.json"):
        with open("./preprocessing/vocab.json", "r") as f:
            vocab = json.load(f)
        return vocab
    vocab = {}
    #For a given key in the vocab, their is a corresponding list. The first element is the "index" of that key, i.e the integer representation
    #of that word, while the second element is the count of that word, or the number of times it appears in the training data
    #The 150 is me randomly picking a number I think will be bigger than any threshold I use
    #to reduce vocabulary size, guaranteeing that "be.v.01" is still in the vocabulary
    vocab[u"be.v.01"] = [0, 150]
    """with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}"""

    for i in tqdm(sg_dict):
        for o in sg_dict[i]["objects"]:
            if "synsets" in o:
                if len(o["synsets"]) > 0:
                    wordnet_synset = o["synsets"][0].lower().strip()
                    if not wordnet_synset in vocab:
                        vocab[wordnet_synset] = [0,0]
                        vocab[wordnet_synset][0] = len(vocab)
                        vocab[wordnet_synset][1] = 1
                    else:
                        vocab[wordnet_synset][1] += 1

            if "attributes" in o:
                for attribute in o["attributes"]:
                    if len(attribute.split()) == 1:
                        if not attribute.lower().strip() in vocab:
                            vocab[attribute.lower().strip()] = [0,0]
                            vocab[attribute.lower().strip()][0] = len(vocab)
                            vocab[attribute.lower().strip()][1] = 1
                        else:
                            vocab[attribute.lower().strip()][1] += 1

        for r in sg_dict[i]["relationships"]:
            predicate = ""
            if "synsets" in r:
                if len(r["synsets"]) > 0:
                    predicate = r["synsets"][0].lower().strip()
            if len(predicate) > 0:
                if not predicate in vocab:
                    vocab[predicate] = [0,0]
                    vocab[predicate][0] = len(vocab)
                    vocab[predicate][1] += 1
                else:
                    vocab[predicate][1] += 1
            #The subject and object should both already be added
            #via the objects loop above

    with open("./vocab.json", "w") as f:
        json.dump(vocab, f)

    return vocab

def pruneVocab(vocab, threshold=10):
    for k,v  in vocab.items():
        if v[1] < threshold:
            del vocab[k]
    with open("./preprocessing/vocab.json", "w") as f:
        json.dump(vocab, f)

def reIndexVocab(vocab):
    i = 0
    for item in vocab:
        vocab[item][0] = i
        i += 1
    with open("./preprocessing/vocab.json", "w") as f:
        json.dump(vocab, f)

#For a single scene graph, return the attribute and relationship triples
def parseSceneGraph(sg, vocab, count_threshold=10):
    attributes = []
    relationships = []
    ob_id_dict = {}
    for o in sg["objects"]:
        name = ""
        if "synsets" in o:
            if len(o["synsets"]) > 0:
                name = o["synsets"][0].lower().strip()
        if len(name) == 0 or name not in vocab:
            continue
        ob_id_dict[o["object_id"]] = name
        if "attributes" in o:
            for attribute in o["attributes"]:
                if len(attribute.split()) == 1 and attribute in vocab:
                    attributes.append((name, u"be.v.01", attribute.lower().strip()))
    for r in sg["relationships"]:
        predicate = ""
        if "synsets" in r:
            if len(r["synsets"]) > 0:
                predicate = r["synsets"][0].lower().strip()
        if len(predicate) == 0 or predicate not in vocab:
            continue
        if r["subject_id"] in ob_id_dict and r["object_id"] in ob_id_dict:
            relationships.append((ob_id_dict[r["subject_id"]], predicate, ob_id_dict[r["object_id"]]))
    return attributes, relationships

def encodeTriple(vocab, subject, predicate, object):
    #This is a dense representation of the sequence 
    dense = [0,0,0]
    dense[0] = vocab[subject][0]
    dense[1] = vocab[predicate][0]
    dense[2] = vocab[object][0]
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

    return graph_def


    #return graph

def smoothAndNormalizeImg(im, r_mean, g_mean, b_mean):
    im[:,:,0] -= r_mean
    im[:,:,1] -= g_mean
    im[:,:,2] -= b_mean
    #im = filters.gaussian_filter(im, 2, mode='nearest')
    return im
 

def imageDataGenerator(path_to_data, image_files, sg_dict, tf_graph, image_means, vocab, chunk_size = 128, which_to_extract = "conv"):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    r_mean = image_means[0]
    g_mean = image_means[1]
    b_mean = image_means[2]

    """with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}"""


    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(tf_graph, input_map={"images" : im_placeholder})
    graph = tf.get_default_graph()

    for group in grouper(image_files, chunk_size):
        attributes_relationships = []
        images = []
        ids = []
        for im_id in group:
            if im_id == None:
                break
            im = Image.open(image_files[im_id])
            im.load()
            im = imresize(im, (224, 224, 3))
            #Imresize casts back to uint8 for some reason
            im = np.array(im, dtype=np.float64)
            #Shouldn't have to do this anymore, since I'm filtering out grayscale images
            #Skip grayscale images
            #if im.shape != (224, 224, 3):
            #    continue
            attributes, relations = parseSceneGraph(sg_dict[im_id], vocab)
            #Shouldn't have to do this anymore, since we're filtering out images
            #that don't have enough information attached to them, all in the name
            #of predictable batch sizes
            #if (len(attributes) + len(relations)) < 10:
            #    continue
            ids.append(im_id)
            images.append(smoothAndNormalizeImg(im, r_mean, g_mean, b_mean))
            attributes.extend(relations)
            encoded_attributes = []
            for a in attributes:
                encoded_attributes.append(encodeTriple(vocab, a[0], a[1], a[2]))
            attributes_relationships.append(encoded_attributes)
        #im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            feed_dict = { im_placeholder:np.array(images, dtype=np.float32) }
            """for f in [n.name for n in tf.get_default_graph().as_graph_def().node]:
                print f"""
            if which_to_extract == "conv":
                feat_tensor = graph.get_tensor_by_name("import/conv5_3/conv5_3:0")
            elif which_to_extract == "fc":
                feat_tensor = graph.get_tensor_by_name("import/fc7_relu:0")
            else:
                print "Error! Invalid feature extraction arg \"{}\"".format(which_to_extract)
                sys.exit(1)
            feats = sess.run(feat_tensor, feed_dict = feed_dict)
        #Reshape convolutional features from 14 x 14 x 512 per image to 196 x 512
        if which_to_extract == "conv":
            feats = np.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
        #feats = np.array(feats, dtype=np.float32)
        #Each of these should contain the full chunk_size elements
        #Except for when the generator runs out of files
        yield feats, ids, attributes_relationships

def checkGrayscale(im_path):
    im = Image.open(im_path)
    im.load()
    im = np.array(im)
    return len(im.shape) == 2

def enoughElements(sg_dict, im_id, vocab):
    attributes, relations = parseSceneGraph(sg_dict[im_id], vocab)
    return (len(attributes) + len(relations)) >= 10

def toNPZ(path_to_data, vgg_tf_model, which_to_extract="conv"):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    path_to_images = path_to_data + "all_images/"
    print "Computing Image Mean"
    r_mean, g_mean, b_mean = computeImageMean(path_to_images)
    image_means = [r_mean, g_mean, b_mean]
    print "Done"

    print "Creating list of files which have a scene graph,\
         excluding grayscale and images with less than 10 combined relations and attributes"
    with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}
    #TODO: Save this sg_dict?

    print "Building vocabulary"
    vocab = buildVocab(sg_dict)
    print "Done"

    print "Pruning vocabulary"
    pruneVocab(vocab)
    reIndexVocab(vocab)
    print "Done"

    print "Size of vocabulary after pruning"
    print len(vocab)

    """image_files_dict_path = "./preprocessing/image_file_list.json"
    if os.path.exists(image_files_dict_path):
        with open(image_files_dict_path, "r") as f:
            image_files = json.load(f)
    else:"""
    #Has to be a dict rather than a list because certain im_ids might not have associated scene graphs
    #If a list were used it could results in list index out of range errors
    #We're filtering out grayscale images
    image_files = {im_id:"{}{}.jpg".format(path_to_images, im_id) \
            for im_id in sorted(sg_dict) if not checkGrayscale("{}{}.jpg".format(path_to_images, im_id))}
    #Also filter out those files which don't have at least ten combined relationships and attributes
    bad = []
    for im_id in image_files:
        if not enoughElements(sg_dict, im_id, vocab):
            bad.append(im_id)
    image_files = {im_id:image_files[im_id] for im_id in image_files if im_id not in bad}
    """#TODO Definitely save this image_files dict
        with open(image_files_dict_path, "w") as f:
            json.dump(image_files, f)"""
    print "Done"

    print "Reading in tensorflow model"
    tf_graph = readInTensorflowModel(vgg_tf_model)
    print "Done"

    print "Creating image generator"
    #Note that any batch significantly larger than 128 might cause a GPU OOM
    #e.g on a 12GB Titan X a batch size of 256 was too big
    batch_size = 128
    im_generator = imageDataGenerator(path_to_data, image_files, sg_dict, tf_graph, image_means, vocab, chunk_size = batch_size, which_to_extract = which_to_extract)
    print "Done"
    count = 0
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    for image_feats, id_batch, att_rels_batch in im_generator:
        path_to_batch_file = "{}batches/batch_{}.npz".format(path_to_data, count)
        if which_to_extract == "conv":
            path_to_batch_file = "{}conv_batches/batch_{}.npz".format(path_to_data, count)
        count += 1
        feat_list = [i for i in image_feats]
        cap_list = [np.array(i) for i in att_rels_batch]
        save_list = [None]*(len(feat_list)+len(cap_list))
        save_list[::2] = feat_list
        save_list[1::2] = cap_list
        np.savez(path_to_batch_file, save_list)
    if which_to_extract == "conv":
        normalizeFeatures(os.path.join(path_to_data, "conv_batches/"), which_to_extract=which_to_extract)
    else:
        normalizeFeatures(os.path.join(path_to_data, "batches/"), which_to_extract=which_to_extract)


if __name__ == "__main__":
    with open("./config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            if line_[0] == "visual_genome":
                path_to_data = line_[1]
            elif line_[0] == "vgg_tf_model":
                vgg_tf_model = line_[1]

    path_to_data += "/" if path_to_data[-1] != "/" else ""

    #vgg_tf_model = "/home/user/misc_github_projects/Scene-Graph-GAN/models/vgg/vgg16.tfmodel"
    #path_to_data = "/home/user/data/visual_genome/"
    toNPZ(path_to_data, vgg_tf_model, which_to_extract="conv")
    #toNPZ(path_to_data, vgg_tf_model, which_to_extract="fc")
