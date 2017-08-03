import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
import os
import PIL.Image as Image
import json
import argparse
from scipy.ndimage import filters
from scipy.misc import imresize
from itertools import izip_longest
from tqdm import tqdm

def computeImageMean(image_dir):
    if os.path.exists("./preprocessing/saved_data/image_mean.txt"):
        means = open("./preprocessing/saved_data/image_mean.txt", "r").read().strip().split()
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

    with open("./preprocessing/saved_data/image_mean.txt", "w") as f:
        f.write("{} {} {}".format(r_mean, g_mean, b_mean))

    return r_mean, g_mean, b_mean

def normalizeFeatures(train_path, batch_dir):
    #Go through all batches once and calculate the min, max, and means
    print "Calculating feature statistics"
    mean = np.zeros((512))
    std_dev = np.zeros((512))
    file_count = 0.0
    for f in os.listdir(train_path):
        if f[-4:] != ".npz":
            continue
        npz = np.load(os.path.join(train_path, f))
        big_arr = npz['arr_0']
        num_ims = 0.0
        temp_mean = np.zeros((512))
        temp_std_dev = np.zeros((512))
        for i in xrange(0, big_arr.shape[0], 3):
            #Im feats is 196 x 512 when conv is selected
            im_feats = big_arr[i]
            #Calculate the per channel mean
            temp_mean = temp_mean + np.mean(im_feats, axis=0)
            #Calculate the per channel std deviation
            temp_std_dev = temp_std_dev + np.std(im_feats, axis=0)
            num_ims += 1.0
        temp_mean = temp_mean / num_ims
        temp_std_dev = temp_std_dev / num_ims
        mean = mean + temp_mean
        std_dev = std_dev + temp_std_dev
        file_count += 1.0
            
    mean = mean / file_count
    std_dev = std_dev / file_count
    print "Done"

    print "Normalizing features"
    #Now center and normalize the data
    for f in os.listdir(batch_dir):
        if f[-4:] != ".npz":
            continue
        npz = np.load(os.path.join(batch_dir, f))
        big_arr = npz['arr_0']
        for i in xrange(0, big_arr.shape[0], 3):
            big_arr[i] -= mean
            big_arr[i] *= (1./std_dev)
        np.savez(os.path.join(batch_dir, f), big_arr)
    print "Done"

#When the vocab is built, the first entry is the index and the second entry is
#a count indicating how many times that word appears in the vocabulary
def buildVocab(sg_dict):
    if os.path.exists("./preprocessing/saved_data/vocab.json"):
        with open("./preprocessing/saved_data/vocab.json", "r") as f:
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

    with open(".preprocessing/saved_data/vocab.json", "w") as f:
        json.dump(vocab, f)

    return vocab

def pruneVocab(vocab, threshold=10):
    for k,v  in vocab.items():
        if v[1] < threshold:
            del vocab[k]
    with open("./preprocessing/saved_data/vocab.json", "w") as f:
        json.dump(vocab, f)

def reIndexVocab(vocab):
    i = 0
    for item in vocab:
        vocab[item][0] = i
        i += 1
    with open("./preprocessing/saved_data/vocab.json", "w") as f:
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

def smoothAndNormalizeImg(im, r_mean, g_mean, b_mean):
    im[:,:,0] -= r_mean
    im[:,:,1] -= g_mean
    im[:,:,2] -= b_mean
    #im = filters.gaussian_filter(im, 2, mode='nearest')
    return im

def imageDataGenerator(path_to_data, image_files, sg_dict, tf_graph, image_means, vocab, chunk_size = 128):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    r_mean = image_means[0]
    g_mean = image_means[1]
    b_mean = image_means[2]

    """with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}"""


    tf.reset_default_graph()
    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(tf_graph, input_map={"images" : im_placeholder})
    graph = tf.get_default_graph()

    for group in grouper(image_files, chunk_size):
        #attributes_relationships = []
        attributes_batch = []
        relations_batch = []
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
            #attributes.extend(relations)
            encoded_attributes = []
            encoded_relations = []
            for a in attributes:
                encoded_attributes.append(encodeTriple(vocab, a[0], a[1], a[2]))
            for r in relations:
                encoded_relations.append(encodeTriple(vocab, r[0], r[1], r[2]))
            #attributes_relationships.append(encoded_attributes)
            attributes_batch.append(encoded_attributes)
            relations_batch.append(encoded_relations)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            feed_dict = { im_placeholder:np.array(images, dtype=np.float32) }
            feat_tensor = graph.get_tensor_by_name("import/conv5_3/conv5_3:0")
            feats = sess.run(feat_tensor, feed_dict = feed_dict)
        #Reshape convolutional features from 14 x 14 x 512 per image to 196 x 512
        feats = np.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
        #feats = np.array(feats, dtype=np.float32)
        #Each of these should contain the full chunk_size elements
        #Except for when the generator runs out of files
        yield feats, ids, attributes_batch, relations_batch

def checkGrayscale(im_path):
    im = Image.open(im_path)
    im.load()
    im = np.array(im)
    return len(im.shape) == 2

def enoughElements(sg_dict, im_id, vocab):
    attributes, relations = parseSceneGraph(sg_dict[im_id], vocab)
    return (len(attributes) > 0 and len(relations) > 0)

def genAndSaveImFeats(path_to_data, path_to_images, image_files, sg_dict, tf_graph, image_means, vocab, batch_size, batch_path, eval = False):
    print "Creating image generator"
    im_generator = imageDataGenerator(path_to_data, image_files, sg_dict, tf_graph, image_means, vocab, chunk_size = batch_size)
    print "Done"
    count = 0
    for image_feats, id_batch, attributes_batch, relations_batch in im_generator:
        path_to_batch_file = os.path.join(batch_path, "batch_{}.npz".format(count))
        feat_list = [i for i in image_feats]
        atts_list = [np.array(i) for i in attributes_batch]
        rels_list = [np.array(i) for i in relations_batch]
        save_list = [None]*(len(feat_list)+len(atts_list)+len(rels_list))
        save_list[::3] = feat_list
        save_list[1::3] = atts_list
        save_list[2::3] = rels_list
        np.savez(path_to_batch_file, save_list)
        #Do some extra steps for the eval step
        if eval and count == 0:
            print "Writing a list of files corresponding to images from eval batch_0"
            #Do this for only the first batch 
            #Filenames order should match with feature order in the save list
            #That way when the batch_0.npz file is opened, big_arr[0] should contain features
            #for the image at filenames[0], big_arr[2] should be for filenames[1], etc
            filenames = [os.path.join(path_to_images, "{}.jpg".format(id)) for id in id_batch]
            path_to_filenames = os.path.join(batch_path, "filenames.txt")
            #Create list of filenames if it doesn't exist
            with open(path_to_filenames, "w") as f:
                for name in filenames:
                    f.write(name + "\n")
            print "Done"
        count += 1


def writeFilenameToFeatDict(eval_path):
    path_to_dict = os.path.join(eval_path, "filename_to_feats_dict.json")
    path_to_filenames = os.path.join(eval_path, "filenames.txt")
    npz = np.load(os.path.join(eval_path, "batch_0.npz"))
    big_arr = npz['arr_0']
    filename_to_feats = {}
    with open(path_to_filenames, "r") as filenames:
        big_arr_index = 0
        for name in filenames:
            filename_to_feats[name.strip()] = big_arr[big_arr_index].tolist()
            big_arr_index += 3
                
    with open(path_to_dict, "w") as dict_file:
        json.dump(filename_to_feats, dict_file)

def toNPZ(path_to_data, path_to_images, out_path, vgg_tf_model):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    path_to_images = os.path.join(path_to_data, "all_images/")
    print "Computing Image Mean"
    #TODO Fix the data snooping (shouldn't matter too much)
    r_mean, g_mean, b_mean = computeImageMean(path_to_images)
    image_means = [r_mean, g_mean, b_mean]
    print "Done"

    print "Creating a list of files which have a scene graph"
    with open(os.path.join(path_to_data, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}
    print "Done"

    print "Building vocabulary"
    vocab = buildVocab(sg_dict)
    print "Done"

    print "Pruning vocabulary"
    pruneVocab(vocab)
    reIndexVocab(vocab)
    print "Done"

    print "Size of vocabulary after pruning"
    print len(vocab)

    image_file_list = os.path.join(os.path.join("preprocessing", "saved_data"), "image_file_list.json")
    if os.path.exists(image_file_list):
        with open(image_file_list, "r") as f:
            image_files = json.load(f)
        #Convert to int
        image_files = {int(i):image_files[i] for i in image_files}
    else:
        print "Now removing grayscale images and images with less than 10 combined relations and attributes"
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
        print "Done"
        with open(image_file_list, "w") as f:
            json.dump(image_files, f)

    print "Splitting into training and eval"
    training_files = dict(image_files.items()[len(image_files)/10:])
    eval_files = dict(image_files.items()[:len(image_files)/10])

    #Sanity check
    assert len(training_files) > len(eval_files)
    print "Done"

    print "Reading in tensorflow model"
    tf_graph = readInTensorflowModel(vgg_tf_model)
    print "Done"

    #Create directory for training batches and eval batches
    train_path = os.path.join(out_path, "train")
    eval_path = os.path.join(out_path, "eval")

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    #Note that any batch significantly larger than 128 might cause a GPU OOM
    #e.g on a 12GB Titan X a batch size of 256 was too big
    batch_size = 128


    print "Generating image feats for eval data"
    genAndSaveImFeats(path_to_data, path_to_images, eval_files, sg_dict, tf_graph, image_means, vocab, batch_size, eval_path, eval = True)

    print "Generating image feats for training data"
    genAndSaveImFeats(path_to_data, path_to_images, training_files, sg_dict, tf_graph, image_means, vocab, batch_size, train_path, eval = False)

    #IMPORTANT!!!! YOU HAVE TO DO THE EVAL PATH NORMALIZATION BEFORE THE TRAINING PATH
    #SINCE THE EVAL NORMALIZATION DEPENDS ON STATISTICS FROM THE TRAINING PATH
    #FEATURES!!!! IF THE TRAINING PATH GETS NORMALIZED FIRST THEN THE FEATURES ARE OFF
    normalizeFeatures(train_path, eval_path)
    normalizeFeatures(train_path, train_path)

    writeFilenameToFeatDict(eval_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--vg_images", default="./data/all_images/", help="The path to the visual genome images. Defaults to ./data/all_images/")
    parser.add_argument("--vg_data", default="./data/batches/", help="The path where you want to save batches. For this script defaults to ./data/lu_et_al_batches")
    parser.add_argument("--vgg_model", default="./models/vgg/vgg16.tfmodel", help="The path to the VGG tensorflow model definition")

    #parser.add_argument("--", default="./", help="")

    args = parser.parse_args()
    params = vars(args)

    #toNPZ(path_to_data, vgg_tf_model)
    toNPZ(params["visual_genome"], params["vg_images"], params["vg_data"], params["vgg_model"])
