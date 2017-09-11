import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
import argparse
import json
import collections
import operator

from train import SceneGraphWGAN

#####################################################
# Global Variables
#####################################################

def init():
    global decoder
    global entity_identifier
    decoder = {}
    entity_identifier = 0

class Entity(object):
    def __init__(self, unique_id, vocab_index):
        self.unique_id = unique_id
        self.og_vocab_index = vocab_index
        self.vocab_indices = {vocab_index : 1}
        self.attributes = []
        #self.duplicates = [self.unique_id]
        #self.type? 

    def addAttribute(self, attribute):
        self.attributes.append(attribute)

    def getVocabIndex(self):
        return collections.Counter(self.vocab_indices).most_common(1)[0][0]

    def getName(self):
        global decoder
        #index = collections.Counter(self.vocab_indices).most_common(1)
        index = sorted(self.vocab_indices.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        return "{}_{}".format(decoder[index], self.unique_id)

    def decodeAttributes(self):
        return [" ".join(self.getName(), "be.v.01", decoder[att.vocab_index]) for att in self.attributes]

class Attribute(object):
    def __init__(self, vocab_index):
        self.vocab_index = vocab_index
    def getVocabIndex(self):
        return self.vocab_index

class Triple(object):
    def __init__(self, triple, attention_vectors, disc_score, is_attribute=True):
        global entity_identifier

        self.is_attribute = is_attribute
        self.attention_vectors = attention_vectors
        self.disc_score = disc_score

        if is_attribute:
            self.subject = Entity(entity_identifier, triple[0])
            entity_identifier += 1
            self.predicate = triple[1]
            self.object = Attribute(triple[2])
            self.subject.addAttribute(self.object)
        else:
            #Otherwise it's a relation
            self.subject = Entity(entity_identifier, triple[0])
            entity_identifier += 1
            self.predicate = triple[1]
            self.object = Entity(entity_identifier, triple[2])
            entity_identifier += 1

        self.triple = [self.subject, self.predicate, self.object]

    def decode(self):
        global decoder
        if self.is_attribute:
            return "{} {} {}".format(self.triple[0].getName(), decoder[self.triple[1]], decoder[self.triple[2].vocab_index])
        else:
            return "{} {} {}".format(self.triple[0].getName(), decoder[self.triple[1]], self.triple[2].getName())


def generatePredictions(wgan, sess):
    #Initialize the queue variables
    queue_vars = [v for v in tf.global_variables() if wgan.queue_var_name in v.name]
    queue_init_op = tf.variables_initializer(queue_vars)
    sess.run(queue_init_op)

    #If the parameters indicate that this is relations and attributes, then the first batch dequeued by the queue runner will be relations, the second attributes
    #See the validationGenerator function in custom runner for proof
    relations, r_score, subject_att, predicate_att, object_att = sess.run([wgan.fake_inputs, wgan.disc_fake, wgan.g.attention_vectors[0], wgan.g.attention_vectors[1], wgan.g.attention_vectors[2]])
    r_attention_vectors = np.transpose(np.array([subject_att, predicate_att, object_att]), (1, 0, 2))
    relations = np.argmax(relations, axis=2)
    rel_triples = [Triple(relations[i], r_attention_vectors[i], r_score[i][0], is_attribute=False) for i in range(relations.shape[0])]
    att_triples = []

    if not wgan.dataset_relations_only:
        queue_vars = [v for v in tf.global_variables() if wgan.queue_var_name in v.name]
        queue_init_op = tf.variables_initializer(queue_vars)
        sess.run(queue_init_op)

        attributes, a_score, a_subject_att, a_predicate_att, a_object_att = sess.run([wgan.fake_inputs, wgan.disc_fake, wgan.g.attention_vectors[0], wgan.g.attention_vectors[1], wgan.g.attention_vectors[2]])
        a_attention_vectors = np.transpose(np.array([a_subject_att, a_predicate_att, a_object_att]), (1, 0, 2))
        attributes = np.argmax(attributes, axis=2)
        att_triples = [Triple(attributes[i], a_attention_vectors[i], a_score[i][0], is_attribute=True) for i in range(attributes.shape[0])]

    att_triples.extend(rel_triples)

    return att_triples
