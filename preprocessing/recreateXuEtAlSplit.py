import h5py
import requests
import json
import argparse

import os, sys
sys.path.append(os.getcwd())

from subprocess import call

from preprocessing.vg_to_imdb import main as create_imdb

#Function to get the region of interest hdf5 file
def downloadROIAndVocab(saved_data_path):
    #Use requests to get it
    link = "http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5"
    r = requests.get(link, stream = True)
    with open("VGG-SGG.h5", 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    predicate_link = "https://raw.githubusercontent.com/danfeiX/scene-graph-TF-release/master/data_tools/VG/predicate_list.txt"
    r = requests.get(predicate_link, stream = True)
    with open(os.path.join(saved_data_path, "predicate_list.txt"), "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    object_link = "https://raw.githubusercontent.com/danfeiX/scene-graph-TF-release/master/data_tools/VG/object_list.txt"
    r = requests.get(object_link, stream = True)
    with open(os.path.join(saved_data_path, "object_list.txt"), "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def createVocabJson(saved_data_path):
    #Objects will have the same index as is present in the h5 file, relations will be the same + 150
    object_f = os.path.join(saved_data_path, "object_list.txt")
    predicate_f = os.path.join(saved_data_path, "predicate_list.txt")
    vocab_f = os.path.join(saved_data_path, "xu_et_al_vocab.json")
    vocab = {}
    #Pretty sure their vocab is 1 indexed
    #Determined this by running min([roi_h5['predicates'][i][0] for i in xrange(len(roi_h5['predicates']))])
    index = 1
    with open(object_f, "r") as f:
        for line in f:
            token = line.strip()
            #The structure here is [index, num_times_appears_in_training_data]
            #The second part of this list is irrelevant, just using to stay consistent with
            #the other preprocessing script
            vocab[token] = [index, 150]
            index += 1
    #Should start at index 151
    with open(predicate_f, "r") as f:
        for line in f:
            token = line.strip()
            vocab[token] = [index, 150]
            index += 1
    with open(vocab_f, "w") as f:
        json.dump(vocab, f)

def readInTensorflowModel(vgg_tf_model):
    #Load VGG Feature Extractor
    with open(vgg_tf_model, mode="rb") as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    return graph_def

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def computeImageMean(image_dir, saved_data_path):
    if os.path.exists(os.path.join(saved_data_path, "image_mean.txt")):
        means = open(os.path.join(saved_data_path, "image_mean.txt"), "r").read().strip().split()
        r_mean = float(means[0])
        g_mean = float(means[1])
        b_mean = float(means[2])
        return r_mean, g_mean, b_mean
    num_images = 0
    r_total = g_total = b_total = 0.0
    r_mean = g_mean = b_mean = 0.0
    for image_file in os.listdir(image_dir):
        #img = cv2.imread(image_dir + image_file).astype(np.float32)
        try:
            img = Image.open(os.path.join(image_dir, image_file))
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

    with open(os.path.join(saved_data_path, "image_mean.txt"), "w") as f:
        f.write("{} {} {}".format(r_mean, g_mean, b_mean))

    return r_mean, g_mean, b_mean

def smoothAndNormalizeImg(im, r_mean, g_mean, b_mean):
    im[:,:,0] -= r_mean
    im[:,:,1] -= g_mean
    im[:,:,2] -= b_mean
    #im = filters.gaussian_filter(im, 2, mode='nearest')
    return im

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
        for i in xrange(0, big_arr.shape[0], 2):
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

def getRelations(index, imbd, roi_h5):
    triples = []
    list_of_rels = roi_h5['relationships'][roi_h5['img_to_first_rel'][index]:roi_h5['img_to_last_rel'][index]]
    for j in range(0, len(list_of_rels)):
        subject = int(roi_h5['labels'][list_of_rels[j][0]])
        #Add 150 to the predicate's index to make it compatible with our restructured vocabulary,
        #which contains both objects and relations
        predicate = int(roi_h5['predicates'][roi_h5['img_to_first_rel'][i] + j]) + 150
        object = int(subject = roi_h5['labels'][list_of_rels[j][1]])
        triple = [subject, predicate, object]
        triples.append(triple)
    return triples

def imageDataGenerator(path_to_images, roi_h5, indices, tf_graph, image_means,chunk_size = 128):
    #Iterate over chunk_sized groups of images, generating features and yielding them
    #along with the attributes and relationships of the image
    r_mean = image_means[0]
    g_mean = image_means[1]
    b_mean = image_means[2]

    tf.reset_default_graph()
    im_placeholder = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(tf_graph, input_map={"images" : im_placeholder})
    graph = tf.get_default_graph()

    for group in grouper(indices, chunk_size):
        relations_batch = []
        images = []
        ids = []
        for index in group:
            if index == None:
                break
            im_id = imdb['image_ids'][index]
            im = Image.open(os.path.join(path_to_images, "{}.jpg".format(im_id)))
            im.load()
            im = imresize(im, (224, 224, 3))
            #Imresize casts back to uint8 for some reason
            im = np.array(im, dtype=np.float64)
            ids.append(im_id)
            images.append(smoothAndNormalizeImg(im, r_mean, g_mean, b_mean))
            relations = getRelations(index, imdb, roi_h5)
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
        yield feats, ids, relations_batch


def genAndSaveImFeats(path_to_images, tf_graph, image_means, batch_size, batch_path, eval = False):
    #Create the database objects
    imdb = h5py.File("imdb_1024.h5", 'r')
    roi_h5 = h5py.File("VG-SGG.h5", 'r')

    if eval:
        #Index 75651 is where the split 0 stops and split 1 starts
        indices = list(range(75651, len(roi_h5['split'])))
    else:
        indices = list(range(0, 75651))

    print "Creating image generator"
    im_generator = imageDataGenerator(path_to_images, imdb, roi_h5, indices, tf_graph, image_means, chunk_size = batch_size)
    print "Done"
    count = 0
    for image_feats, id_batch, relations_batch in im_generator:
        path_to_batch_file = os.path.join(batch_path, "batch_{}.npz".format(count))
        feat_list = [i for i in image_feats]
        rels_list = [np.array(i) for i in relations_batch]
        save_list = [None]*(len(feat_list)+len(rels_list))
        save_list[::2] = feat_list
        save_list[1::2] = rels_list
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

def toNPZ(params):
    print "Reading in tensorflow model"
    tf_graph = readInTensorflowModel(params["vgg_model"])
    print "Done"

    print "Computing Image Mean"
    #TODO Fix the data snooping (shouldn't matter too much)
    r_mean, g_mean, b_mean = computeImageMean(params["vg_images"], params["saved_data"])
    image_means = [r_mean, g_mean, b_mean]
    print "Done"

    #Create directory for training batches and eval batches
    train_path = os.path.join(params["vg_batches"], "train")
    eval_path = os.path.join(params["vg_batches"], "eval")

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    print "Generating image feats for eval data"
    genAndSaveImFeats(params["vg_images"], tf_graph, image_means, params["batch_size"], eval_path, eval = True)

    print "Generating image feats for training data"
    genAndSaveImFeats(params["vg_images"], train_path, eval = False)

    #IMPORTANT!!!! YOU HAVE TO DO THE EVAL PATH NORMALIZATION BEFORE THE TRAINING PATH
    #SINCE THE EVAL NORMALIZATION DEPENDS ON STATISTICS FROM THE TRAINING PATH
    #FEATURES!!!! IF THE TRAINING PATH GETS NORMALIZED FIRST THEN THE FEATURES ARE OFF
    normalizeFeatures(train_path, eval_path)
    normalizeFeatures(train_path, train_path)

    writeFilenameToFeatDict(eval_path)


def main(args, params):
    create_imdb(args)
    #Download necessary stuff
    downloadROIAndVocab(params["saved_data"])
    #Create the vocabulary
    createVocabJson(params["saved_data"])
    #Convert the now constructed hdf5 dataset to our npz files
    toNPZ(params)
    #Remove their files
    call(["rm", "imdb_1024.h5"])
    call(["rm", "VG-SGG.h5"])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--vg_images", default="./data/all_images/", help="The path to the visual genome images. Defaults to ./data/all_images/")
    parser.add_argument("--vg_batches", default="./data/xu_et_al_batches/", help="The path where you want to save batches. For this script defaults to ./data/lu_et_al_batches")
    parser.add_argument("--vgg_model", default="./models/vgg16.tfmodel", help="The path to the VGG tensorflow model definition")
    parser.add_argument("--batch_size", default=129, type=int, help="The batch size. Note that batch sizes above 128 have the potential to cause OOM errors")
    parser.add_argument("--saved_data", default="./preprocessing/saved_data/", help="The path to save various data for calculating image means, vocabularies, etc.")

    #These args are necessary for running their code
    parser.add_argument('--image_dir', default='./data/all_images')
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--imh5_dir', default='.')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--metadata_input', default='./data/image_data.json', type=str)

    args = parser.parse_args()
    params = vars(args)

    if not os.path.exists(params["saved_data"]):
        os.makedirs(params["saved_data"])

    #Yeah I know this is stupid but I'm lazy
    main(args, params)
