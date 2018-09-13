import os, sys
sys.path.append(os.getcwd())

import argparse

from grab_data import getVisualGenome, streamSaveLink
from arrange_data import addAttributes, unzipAll, moveAll
from calculate_image_mean import computeImageStats
from map_files_to_triples import createVocab, loadWordEmbeddings, mapFromImagesToTriples

if __name__ == "__main__":

    ##################################################
    ## Argparse stuff                                #
    ##################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--add_attributes", type=bool, default=True, help="Whether or not to add attributes to the scene graphs")
    parser.add_argument("--path_to_artifacts", default="./dataset_creation", help="Path to store things like vocab, image means, etc")
    parser.add_argument("--path_to_word_embeddings", default="./data/GoogleNews-vectors-negative300.bin", help="Path to the binary file of the word embeddings")

    args = parser.parse_args()
    params = vars(args)

    #if not os.path.exists(params["visual_genome"]):
    #    os.makedirs(params["visual_genome"])

    #print "Downloading"
    #getVisualGenome(params["visual_genome"])
    #print "Unzipping"
    #unzipAll(params["visual_genome"])
    #print "Rearranging"
    #moveAll(params["visual_genome"])

    #print "Adding attributes"
    #if params["add_attributes"]: 
    #    addAttributes(params["visual_genome"])

    #print "Computing image stats"
    #computeImageStats(os.path.join(params["visual_genome"], "all_images"), params["path_to_artifacts"])

    path_to_sgs = os.path.join(params["visual_genome"], "scene_graphs.json")
    path_to_vocab = os.path.join(params["path_to_artifacts"], "vocab.json")
    path_to_maps = os.path.join(params["path_to_artifacts"], "ims_to_triples.json")
    path_to_images = os.path.join(params["visual_genome"], "all_images")
    path_to_word_embeddings_out = os.path.join(params["path_to_artifacts"], "word_embeddings.npy")

    print "Creating Vocab"
    vocab = createVocab(path_to_sgs, path_to_vocab)
    print "Creating Word Embeddings"
    word_embedddings = loadWordEmbeddings(params["path_to_word_embeddings"], path_to_word_embeddings_out, vocab)
    print "Creating map"
    mapFromImagesToTriples(vocab, path_to_images, path_to_sgs, path_to_maps)
