import os, sys
sys.path.append(os.getcwd())

import json
import argparse

from tqdm import tqdm

def createVocab(path_to_scene_graphs, vocab_out_path):
    with open(path_to_scene_graphs, "r") as f:
        sgs = json.load(f)

    threshold = 50
    vocab = {}
    counts = {}
    vocab["be"] = 0
    counts["be"] = threshold + 1

    for sg in tqdm(sgs, total=len(sgs)):
        #Collect vocab of atts
        for a in sg['attributes']:
            if 'attributes' in a['attribute']:
                for a_name in a['attribute']['attributes']:
                    cleaned = a_name.lower().strip()
                    if cleaned not in vocab:
                        vocab[cleaned] = len(vocab)
                        counts[cleaned] = 0
                    else:
                        counts[cleaned] += 1
        #Collect vocab of objects
        for o in sg['objects']:
            for name in o["names"]:
                cleaned = name.lower().strip()
                if cleaned not in vocab:
                    vocab[cleaned] = len(vocab)
                    counts[cleaned] = 0
                else:
                    counts[cleaned] += 1

        #Collect vocab of relations
        for r in sg['relationships']:
            cleaned = r["predicate"].lower().strip()
            if cleaned not in vocab:
                vocab[cleaned] = len(vocab)
                counts[cleaned] = 0
            else:
                counts[cleaned] += 1

    for k in vocab.keys():
        if counts[k] < threshold:
            del vocab[k]

    #Re-index
    vocab = {k:i for i, k in enumerate(vocab.keys())}

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
