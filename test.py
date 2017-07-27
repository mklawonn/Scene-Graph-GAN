import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np

import json
import argparse

from train import SceneGraphWGAN

attributes_flag = 0.0
relations_flag = 1.0

vocab_size = 0

def loadModel(sess, params):
    #Create WGAN instance
    wgan = SceneGraphWGAN(params["visual_genome"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"], resume=False)
    wgan.constructOps()
    wgan.loadModel(sess)
    return tf.get_default_graph().as_graph_def()

def generatePredictions(image_features, graph_def, sess, relations_only = True):
    #Enqueue image features
    #Initialize the queue variables
    dummy_triples = np.zeros((image_features.shape[0], 3, vocab_size), dtype=np.float32)
    if relations_only:
        feed_dict = {im_feats_placeholder : image_features, triples_placeholder : dummy_triples, flag_placeholder : relations_flag}
        #Generate a whole bunch of triples
        #TODO: If the Add_14 tensor gets renamed make sure to change it here
        triples_tensor = graph_def.get_tensor_by_name("Generator_1/generator_output:0")
        scores_tensor = graph_def.get_tensor_by_name("Discriminator_2/Add_14:0")
        
        sub_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax:0")
        pred_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax_1:0")
        obj_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax_2:0")

        triples, scores, sub_att, pred_att, obj_att = sess.run([triples_tensor, scores_tensor, sub_att_vector, pred_att_vector, obj_att_vector], feed_dict=feed_dict)
    else:
        pass
    #Stitch together using attention
    #Make sure to rename objects that are the same name but different according to the attention
    #Return the argmaxed triples
    return np.argmax(triples, axis=2), np.mean(scores, axis=1)

def loadAllValidationImages(path_to_val_batches):
    filenames = [os.path.join(path_to_val_batches, f) for f in os.listdir(path_to_val_batches) if f[-4:] == ".npz"]
    big_arr_list = []
    for f in range(len(filenames)):
        npz = np.load(filenames[f])
        big_arr_list.append(npz['arr_0'])
    return np.concatenate(big_arr_list, axis=0)

def recallAtK(im_feats, ground_truth_attributes, ground_truth_relations, model_def, sess, k, batch_size):
    #Need to tile image features to be the batch_size that the model was built with
    im_feats = np.reshape(im_feats, (1, im_feats.shape[0], im_feats.shape[1]))
    im_feats = np.tile(im_feats, (batch_size, 1, 1))
    fake_relations, scores = generatePredictions(im_feats, model_def, sess)
    #Filter the fake relations based on associated scores, keeping only the top k
    #Argsort scores
    indices = np.argsort(scores)
    #Reverse order
    indices = np.flip(indices, axis=0)
    #Truncate index list to top k
    indices = indices[k:]
    #Cut out any fake_relations not in the top k
    fake_relations = fake_relations[indices]

    #Compute the intersection of the fake_relations and the ground truth relations
    #TODO: CRITICAL! We need to change duplicate entries in the ground_truth such that they're not identical
    #e.g if there are two man-wears-shirt entries, change one to man1-wears-shirt1
    #but then you probably also want to make sure that if both the fake and the ground truth describe the same duplicates,
    #they also name them in the same way. E.g make sure man1 in gt isn't man in predicted
    gt_relations_set = set([tuple(i) for i in ground_truth_relations.tolist()])
    fake_relations_set = set([tuple(i) for i in fake_relations.tolist()])

    intersection_size = float(len(gt_relations_set.intersection(fake_relations_set)))

    #Divide by the number of ground_truth relations
    num_relations = float(ground_truth_relations.shape[0])
    r_at_k = intersection_size / num_relations
    return r_at_k
    

def measurePerformance(model_def, big_arr, sess, batch_size):
    recallAtKDict = {50 : [], 100 : []}
    for k in recallAtKDict:
        for i in xrange(0, big_arr.shape[0], 3):
            recallAtKDict[k].append(recallAtK(big_arr[i], big_arr[i+1], big_arr[i+2], model_def, sess, k, batch_size))
        recallAtKDict[k] = (float(sum(recallAtKDict[k])) / float(len(recallAtKDict[k])))*100.0
    return recallAtKDict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--GPU", default="0", help="Which GPU to use")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/vocab.json", help="Path to the vocabulary")

    parser.add_argument("--batch_size", default=256, help="Batch size defaults to 256", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of iterations to train the critic", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--epochs", default=30, help="Number of epochs defaults to 30", type=int)
    parser.add_argument("--print_interval", default=500, help="The model will be saved and samples will be generated every <print_interval> iterations", type=int)
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")
    parser.add_argument("--lambda", default=10, help="Lambda term which regularizes to be close to one lipschitz", type=int)
    parser.add_argument("--use_language", default=False, help="Determines whether the generator update is also based on a discriminator trained on language only", type=bool)


    args = parser.parse_args()
    params = vars(args)

    with open(params["vocab"], "r") as f:
        vocab = json.load(f)
        vocab_size = len(vocab)

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = params["GPU"]

    #Call this just in case the graph is persisting due to TF closing suddenly
    tf.reset_default_graph()

    path_to_val_batches = os.path.join(params["visual_genome"], "eval")
    
    big_arr = loadAllValidationImages(path_to_val_batches)
    with tf.Session() as sess:
        model_def = loadModel(sess, params)
        recallAtKDict = measurePerformance(model_def, big_arr, sess)
        for k in recallAtKDict:
            print "Recall @ {}: {}".format(k, recallAtKDict[k])
