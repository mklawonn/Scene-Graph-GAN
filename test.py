import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np

import json
import argparse
import time
from tqdm import tqdm

from train import SceneGraphWGAN

from postprocessing.generate_samples import generatePredictions
from postprocessing.construct_graph import *

attributes_flag = 0.0
relations_flag = 1.0

vocab_size = 0

def recallAtK(generated_triples, ground_truth_relations, k, batch_size):

    #I think we need to avoid altering generated triples in place
    sorted_triples = generated_triples[:]
    sorted_triples.sort(key=lambda x : x.disc_score, reverse=True)
    sorted_triples = sorted_triples[:k]
    
    gt_relations_set = set([tuple(i) for i in ground_truth_relations.tolist()])

    fake_relations_set = set([(i.triple[0].getVocabIndex(), i.triple[1], i.triple[2].getVocabIndex()) for i in sorted_triples])

    relations_intersection = float(len(gt_relations_set.intersection(fake_relations_set)))

    #Divide by the number of ground_truth relations
    num_relations = float(ground_truth_relations.shape[0])
    r_at_k = relations_intersection / num_relations
    return r_at_k

def recallAtKWithAttributes(generated_triples, ground_truth_relations, ground_truth_attributes, k, batch_size):
    ground_truth_triples = np.concatenate((ground_truth_relations, ground_truth_attributes), axis=0)

    generated_relations = [i for i in generated_triples if not i.is_attribute]
    generated_attributes = [i for i in generated_triples if i.is_attribute]

    sorted_relations = generated_relations[:]
    sorted_relations.sort(key=lambda x : x.disc_score, reverse=True)
    sorted_relations = sorted_relations[:k]

    sorted_attributes = generated_attributes[:]
    sorted_attributes.sort(key=lambda x : x.disc_score, reverse=True)
    sorted_attributes = sorted_attributes[:k]

    sorted_triples = generated_triples[:]
    sorted_triples.sort(key=lambda x : x.disc_score, reverse=True)
    sorted_triples = sorted_triples[:k]

    gt_relations_set = set([tuple(i) for i in ground_truth_relations.tolist()])
    gt_attributes_set = set([tuple(i) for i in ground_truth_attributes.tolist()])
    gt_triples_set = set([tuple(i) for i in ground_truth_triples.tolist()])

    fake_relations_set = set([(i.triple[0].getVocabIndex(), i.triple[1], i.triple[2].getVocabIndex()) for i in sorted_relations])
    fake_attributes_set = set([(i.triple[0].getVocabIndex(), i.triple[1], i.triple[2].getVocabIndex()) for i in sorted_attributes])
    fake_triples_set = set([(i.triple[0].getVocabIndex(), i.triple[1], i.triple[2].getVocabIndex()) for i in sorted_triples])

    relations_intersection = float(len(gt_relations_set.intersection(fake_relations_set)))
    attributes_intersection = float(len(gt_attributes_set.intersection(fake_attributes_set)))
    triples_intersection  = float(len(gt_triples_set.intersection(fake_triples_set)))

    num_relations = float(ground_truth_relations.shape[0])
    num_attributes = float(ground_truth_attributes.shape[0])
    num_triples = num_relations+num_attributes

    rel_r_at_k = relations_intersection / num_relations
    att_r_at_k = attributes_intersection / num_attributes
    triples_r_at_k = triples_intersection / num_triples

    return rel_r_at_k, att_r_at_k, triples_r_at_k

def loadModel(params, sess):
    #Create WGAN instance
    wgan = SceneGraphWGAN(params["vg_batches"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"],
           validation = True, resume=False, dataset_relations_only = params["dataset_relations_only"])
    wgan.constructOps()
    wgan.loadModel(sess)
    decoder = {y:x for x,y in wgan.vocab.iteritems()}
    queue_var_name = wgan.queue_var_name
    #return tf.get_default_graph().as_graph_def()
    tf.train.start_queue_runners(sess=sess)
    wgan.custom_runner.start_threads(sess) 
    #wgan.custom_runner.start_threads(sess)
    return wgan

def printProgress(params, recallAtKDict):
    if params["dataset_relations_only"]:
        for k in recallAtKDict:
            recall = (float(sum(recallAtKDict[k])) / float(len(recallAtKDict[k])))*100.0
            print "This data was relations only: all triples recall at {} was {}".format(k, recall)
    else:
        for k in recallAtKDict:
            #recallAtKDict[k][0] = (float(sum([i[0] for i in recallAtKDict[k]])) / float(len(recallAtKDict[k])))*100.0
            rel_recall = (float(sum([i[0] for i in recallAtKDict[k]])) / float(len(recallAtKDict[k]))) * 100.0
            att_recall = (float(sum([i[1] for i in recallAtKDict[k]])) / float(len(recallAtKDict[k])))*100.0
            trip_recall = (float(sum([i[2] for i in recallAtKDict[k]])) / float(len(recallAtKDict[k])))*100.0
            print "Relations recall at {} was {}".format(k, rel_recall)
            print "Attributes recall at {} was {}".format(k, att_recall)
            print "All triples recall at {} was {}".format(k, trip_recall)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vg_batches", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--GPU", default="0", help="Which GPU to use")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/saved_data/vocab.json", help="Path to the vocabulary")
    parser.add_argument("--critic_iters", default=10)

    parser.add_argument("--batch_size", default=256, help="Batch size defaults to 256", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")
    parser.add_argument("--lambda", default=10, help="Lambda term which regularizes to be close to one lipschitz", type=int)
    parser.add_argument("--use_language", default=False, help="Determines whether the generator update is also based on a discriminator trained on language only", type=bool)
    parser.add_argument("--dataset_relations_only", default=False, help="When true, indicates that the data only contains relations, and will affect how data is read", type=bool)


    args = parser.parse_args()
    params = vars(args)

    with open(params["vocab"], "r") as f:
        vocab = json.load(f)
        vocab_size = len(vocab)

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = params["GPU"]

    #Call this just in case the graph is persisting due to TF closing suddenly
    tf.reset_default_graph()

    recallAtKDict = {50 : [], 100 : []}

    with tf.Session() as sess:
        wgan = loadModel(params, sess)
        print "Done loading, sleeping for some amount of time to allow the queue to populate"
        time.sleep(150)
        count = 0
        for i in tqdm(range(wgan.custom_runner.num_validation_images)):
            triples = generatePredictions(wgan, sess)
            all_entities = findAllEntities(triples)
            potential_duplicates = determinePotentialDuplicates(list(all_entities))
            resolveDuplicateEntities(potential_duplicates, all_entities, triples)

            #TODO Rewrite recall at k and measure performance functions to compare the following two
            #ground truth things to the triples. By this line, triples should be comparable (e.g resolved)
            #Will need to account for the appended id though (e.g person3 should just be person)
            gt_rels = wgan.custom_runner.gt_rels.get()
            gt_atts = wgan.custom_runner.gt_atts.get()
            for k in recallAtKDict:
                if params["dataset_relations_only"]:
                    recallAtKDict[k].append(recallAtK(triples, gt_rels, gt_atts, k, params["batch_size"]))
                else:
                    rel_r_at_k, att_r_at_k, triples_r_at_k = recallAtKWithAttributes(triples, gt_rels, gt_atts, k, params["batch_size"])
                    recallAtKDict[k].append((rel_r_at_k, att_r_at_k, triples_r_at_k))
            del triples
            del all_entities
            del potential_duplicates
            count += 1
            if (count % 500) == 499:
                printProgress(params, recallAtKDict)

    print "-"*35
    print "Final stats"
    print "-"*35
    printProgress(params, recallAtKDict)
