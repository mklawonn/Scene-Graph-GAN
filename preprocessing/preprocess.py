import local as vg
import tensorflow as tf
import numpy as np
import os
import Image
import queue
from scipy.ndimage import filters
from scipy.misc import imresize
from itertools import izip_longest
from tqdm import tqdm

#######################################################################################################
#Function to read in visual genome data and return in manageable batches
#Inputs: Image Directory, path to directory containing scene graphs per image.
#        See local.SaveSceneGraphsById for information on how to save only the scene graph
#        of a particular image.
#Outputs: Yields a batch of K images + associated list of attributes + associated list of relationships
#until N total images have been yielded
#######################################################################################################

def buildVocab(path_to_data):
    to_index_dict = {}
    to_index_dict["is"] = 0
    path_to_data += "/" if path_to_data[-1] != "/" else ""

    scene_graphs = GetSceneGraphs(startIndex=0, endIndex=-1,
                   dataDir=path_to_data, imageDataDir="{}by-id/".format(path_to_data),
                   minRels=0, maxRels=100)
    for g in scene_graphs:
        for i in g.relationships:
            if str(i.subject).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            if str(i.predicate).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            if str(i.object).lower() not in to_index_dict.keys():
                to_index_dict[str(i.object).lower()] = len(to_index_dict.keys())
        for o in graph.objects:
            if str(o).lower() not in to_index_dict.keys():
                to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
            for attribute in o.attributes:
                if str(attribute).lower() not in to_index_dict.keys():
                    to_index_dict[str(i.subject).lower()] = len(to_index_dict.keys())
    return to_index_dict

#TODO: One-hot (or dense one-hot) encoding of triple
def encodeTriple(vocab, triple):
    

def computeImageMean(image_dir):
    num_images = 0
    r_total = g_total = b_total = 0.0
    r_mean = g_mean = b_mean = 0.0
    for image_file in os.listdir(image_dir):
        #img = cv2.imread(image_dir + image_file).astype(np.float32)
        img = np.array(Image.open(image_file))
        img = imresize(img, (224, 224, 3))
        r_total += np.sum(img[:,:,0]) / (img[:,:,0].shape[0] * img[:,:,0].shape[1])
        g_total += np.sum(img[:,:,1]) / (img[:,:,1].shape[0] * img[:,:,1].shape[1])
        b_total += np.sum(img[:,:,2]) / (img[:,:,2].shape[0] * img[:,:,2].shape[1])
        num_images += 1

    r_mean = r_total / float(num_images)
    g_mean = g_total / float(num_images)
    b_mean = b_total / float(num_images)

    return r_mean, g_mean, b_mean

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def imageDataGenerator(path_to_data, vgg_tf_model, chunk_size = 2500):
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    path_to_images = path_to_data + "all_images/"
    image_files = ["{}{}.jpg".format(path_to_images, im_path) \
            for im_path in sorted([int(i.replace(".jpg", "")) for i in os.listdir(path_to_images)])]
    #For preprocessing
    r_mean, g_mean, b_mean = computeImageMean(path_to_images)

    #Load VGG Feature Extractor
    with open(vgg_tf_model, mode="rb") as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={"images" : im_placeholder)

    graph = tf.get_default_graph()
    
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    for group in grouper(image_files, chunk_size):
        ids = []
        attributes_relationships = []
        for f in group:
            if f == None:
                break
            ids.append(int(f[f.rfind("/")+1:-4]))
            graph = vg.GetSceneGraph(ids[-1], images=path_to_images, imageDataDir="{}by-id/".format(path_to_data)):
            #First add the relationships
            attributes_relationships = ["{} {} {}".format(str(i.subject), str(i.predicate).lower(), str(i.object)) \
                             for i in graph.relationships]
            for o in graph.objects:
                for attribute in o.attributes:
                    attributes_relationships.append("{} is {}".format(o, attribute))
            im = np.array(Image.open(f), dtype=np.float32)
            im = imresize(im, (224, 224, 3))
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
        #TODO: Convert attributes_relations into one-hot encoding
        yield feats, ids, attributes_relations


#Function to convert images and associated data into TF Record format
#Inputs: A list of images and attributes. Each tuple in the list is
#        expected to contain three items. The numpy array representing 
#        the image features comes first, then the attributes, and then
#        the relationships.
#Outputs: Outputs a TFRecord 

def toTFRecord(path_to_data, vgg_tf_model):
    im_generator = imageDataGenerator(path_to_data, vgg_tf_model)
    for image_feats, id_batch, att_rels_batch in tqdm(im_generator):
        #Construct the Example proto object
        example = tf.train.Example(
            features=tf.train.Features(
                #A feature contains one of either a int64_list, float list
                #or bytes_list
                feature={
                    'atts_and_rels': tf.train.Feature(
                        ),
                    'image': tf.train.Feature(
                        ),
        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    return


if __name__ == "__main__":
    vgg_tf_model = "/home/user/misc_github_projects/Scene-Graph-GAN/models/vgg/vgg16.tfmodel"
    path_to_data = "/home/user/data/visual_genome/"
    toTFRecord(path_to_data, vgg_tf_model)
