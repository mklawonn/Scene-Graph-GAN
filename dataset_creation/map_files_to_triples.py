import os, sys
sys.path.append(os.getcwd())

import json
import argparse
import struct
import numpy as np
import copy
import operator

from tqdm import tqdm

#Inspired by https://gist.github.com/ottokart/673d82402ad44e69df85
def loadWordEmbeddings(path_to_word_embeddings, word_embeddings_out_path, word_vocab):
    with open(path_to_word_embeddings, "rb") as f:
        #Read in header
        header = ""
        c = None
        while c != "\n":
            c = f.read(1)
            header += c

        #word_embeddings_matrix = np.zeros((len(word_vocab), int(header.split()[1])), dtype=np.float64)
        word_embeddings_matrix = np.random.uniform(low = -0.1, high = 0.1, size=(len(word_vocab), int(header.split()[1])))
        #print len(word_vocab)
        #print max([y for x, y in word_vocab.iteritems()])
        
        counter = 0
        for i in xrange(int(header.split()[0])):
            word = ""
            while True:
                c = f.read(1)
                if c == " ":
                    break
                word += c
            if word not in word_vocab:
                f.read(4*word_embeddings_matrix.shape[1])
                continue
            #The correct index will be determined by the word vocabulary
            #The "4" comes from floats being 4 bytes long
            binary_vector = f.read(4*word_embeddings_matrix.shape[1])
            word_embeddings_matrix[word_vocab[word]] = [struct.unpack_from('f', binary_vector, j)[0]\
                                                for j in xrange(0, len(binary_vector), 4)]
            #Break out of the loop once all the words you're looking for
            #have been found.
            counter += 1
            if counter == word_embeddings_matrix.shape[0]:
                break
    #zeros_vec = np.zeros((int(header.split()[1])), dtype=np.float64)
    ##Remove all entries in the matrix that were never touched
    #word_embeddings_matrix = word_embeddings_matrix[np.where(word_embeddings_matrix != zeros_vec)].reshape((-1, int(header.split()[1]))) 
    np.save(word_embeddings_out_path, word_embeddings_matrix)
    return word_embeddings_matrix


def createVocab(path_to_scene_graphs, vocab_out_path):
    with open(path_to_scene_graphs, "r") as f:
        sgs = json.load(f)

    attribute_counts = {}
    relation_counts = {}
    object_counts = {}

    vocab = {}

    top_n = 150

    for sg in tqdm(sgs, total=len(sgs)):
        #Collect vocab of atts
        for a in sg['attributes']:
            if 'attributes' in a['attribute']:
                for a_name in a['attribute']['attributes']:
                    cleaned = a_name.lower().strip()
                    if cleaned not in attribute_counts:
                        attribute_counts[cleaned] = 1
                    else:
                        attribute_counts[cleaned] += 1
        #Collect vocab of objects
        for o in sg['objects']:
            for name in o["names"]:
                cleaned = name.lower().strip()
                if cleaned not in object_counts:
                    object_counts[cleaned] = 1
                else:
                    object_counts[cleaned] += 1

        #Collect vocab of relations
        for r in sg['relationships']:
            cleaned = r["predicate"].lower().strip()
            if cleaned not in relation_counts:
                relation_counts[cleaned] = 1
            else:
                relation_counts[cleaned] += 1

    top_attributes = sorted(attribute_counts.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    top_objects = sorted(object_counts.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    top_relations = sorted(relation_counts.items(), key=operator.itemgetter(1), reverse=True)[:top_n]

    for v in [top_attributes, top_objects, top_relations]:
        for tup in v:
            if tup[0] in vocab:
                continue
            else:
                vocab[tup[0]] = len(vocab)

    vocab["be"] = len(vocab)

    with open(vocab_out_path, "w") as f:
        json.dump(vocab, f)

    return vocab

def getTriples(sg, vocab):
    os = sg['objects']
    id_to_obj = {o['object_id']:o for o in os}

    triples = []

    for a in sg['attributes']:
        if 'attributes' in a['attribute']:
            for subj_name in id_to_obj[a['attribute']['object_id']]['names']:
                subj = subj_name.lower().strip()
                if subj not in vocab:
                    continue
                for a_name in a['attribute']['attributes']:
                    obj = a_name.lower().strip()
                    if obj not in vocab:
                        continue
                    triples.append(encodeTriple(vocab, [subj, "be", obj]))

    for r in sg['relationships']:
        pred = r['predicate'].lower().strip()
        if pred not in vocab:
            continue
        for subj_name in id_to_obj[r['subject_id']]['names']:
            subj = subj_name.lower().strip()
            if subj not in vocab:
                continue
            for obj_name in id_to_obj[r['object_id']]['names']:
                obj = obj_name.lower().strip()
                if obj not in vocab:
                    continue
                triples.append(encodeTriple(vocab, [subj, pred, obj]))

    return triples
    
def encodeTriple(vocab, triple):
    encoded = [0, 0, 0]
    encoded[0] = vocab[triple[0]]
    encoded[1] = vocab[triple[1]]
    encoded[2] = vocab[triple[2]]
    return encoded

def mapFromImagesToTriples(vocab, path_to_images, path_to_scene_graphs, map_out_path):
    ims_to_triples = {}

    with open(path_to_scene_graphs, "r") as f:
        sgs = json.load(f)

    for sg in tqdm(sgs, total=len(sgs)):
        im_path = os.path.join(path_to_images, "{}.jpg".format(sg['image_id']))
        ims_to_triples[im_path] = getTriples(sg, vocab)
    
    with open(map_out_path, "w") as f:
        json.dump(ims_to_triples, f)
