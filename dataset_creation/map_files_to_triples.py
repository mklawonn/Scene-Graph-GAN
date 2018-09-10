import os, sys
sys.path.append(os.getcwd())

import json
import argparse

def createVocab(path_to_scene_graphs, vocab_out_path):
    vocab = {}
    for sg in sgs:
        #Collect vocab of atts
        for a in sg['attributes']:
            if 'attributes' in a['attribute']:
                for a_name in a['attribute']:
                    cleaned = a_name.lower().strip()
                    if cleaned not in vocab:
                        vocab[cleaned] = len(vocab)
        #Collect vocab of relations
        for o in sg['objects']:
            cleaned = o["name"].lower().strip()
            if cleaned not in vocab:
                vocab[cleaned] = len(vocab)

        for r in sg['relationships']:
            cleaned = r["predicate"].lower().strip()
            if cleaned not in vocab:
                vocab[cleaned] = len(vocab)

    with open(vocab_out_path, "w") as f:
        json.dump(vocab, f)

    return vocab

    

def mapFromImagesToTriples(vocab, path_to_data, path_to_scene_graphs, map_out_path):
    ims_to_triples = {}

    with open(path_to_scene_graphs, "r") as f:
        sgs = json.load(f)

    
    with open(map_out_path, "w") as f:
        json.dump(ims_to_triples)
