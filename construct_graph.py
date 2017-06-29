import json
import itertools

import numpy as np
import tensorflow as tf

attributes_flag = 0.0
relations_flag = 1.0

vocab = {}
decoder = {}
entity_identifier = 0

class Entity(object):
    def __init__(self, unique_id, vocab_index):
        self.unique_id = unique_id
        self.vocab_index = vocab_index
        self.attributes = []
        self.duplicates = [self.unique_id]
        #self.type? 

    def addAttribute(self, attribute):
        self.attributes.append(attribute)

    def decodeAttributes(self):
        return [" ".join(decoder[self.vocab_index], "be.v.01", decoder[att.vocab_index]) for att in self.attributes]

class Attribute(object):
    def __init__(self, vocab_index):
        self.vocab_index = vocab_index

class Triple(object):
    def __init__(self, triple, attention_vector, disc_score, is_attribute=True):
        global entity_identifier

        self.is_attribute = is_attribute
        #self.triple = triple
        self.attention_vector = attention_vector
        self.disc_score = disc_score

        if is_attribute:
            self.subject = Entity(entity_identifier, triple[0])
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
        return " ".join(decoder[self.subject.vocab_index], decoder[self.predicate], decoder[self.object.vocab_index])

class Graph(object):
    def __init__(self, ):
        self.nodes = []

#Function to read in the model
def loadModel(path_to_checkpoints):
    sess = tf.Session()
    checkpoint = tf.train.get_checkpoint_state(path_to_checkpoints)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint.model_checkpoint_path))
        saver.restore(sess, tf.train.latest_checkpoint(path_to_checkpoints))
    else:
        print "ERROR: Could not load model"
        exit(1)
    graph = tf.get_default_graph()
    return sess, graph

def readVocab():
    global vocab
    global decoder
    with open("preprocessing/vocab.json", "r") as f:
        vocab = json.load(f)
    decoder = {y[0]:x for x, y in vocab.iteritems()}

#Function to load filename to feature dict for later use
def getFilenameToFeatDict(batch_path):
    #For image in list in eval
    eval_image_path = os.path.join(batch_path, "eval")
    #Load the filename to feats dictionary
    path_to_dict = os.path.join(eval_image_path, "filename_to_feats_dict.json")
    with open(path_to_dict, 'r') as f:
        filename_to_feats = json.load(f)
    with open(os.path.join(eval_image_path, "filenames.txt"), 'r') as f:
        filenames = [line.strip() for line in f]
    return filename_to_feats, filenames

#Function to generate a bunch of triples (attributes and relations) along with the attention
#vector associated with each subject, predicate, and object
def generateTriples(sess, graph, image_feats, num_per_image = 75):

    generator_output = graph.get_tensor_by_name("Generator_1/transpose:0")
    discriminator_output = graph.get_tensor_by_name("Discriminator_2/transpose:0")
    attention_1 = graph.get_tensor_by_name("Generator_1/RelaxedOneHotCategorical_1/sample/Reshape:0")
    attention_2 = graph.get_tensor_by_name("Generator_1/RelaxedOneHotCategorical_5/sample/Reshape:0")
    attention_3 = graph.get_tensor_by_name("Generator_1/RelaxedOneHotCategorical_9/sample/Reshape:0")

    #real_inputs = graph.get_tensor_by_name("Placeholder:0")
    image_feats_placeholder = graph.get_tensor_by_name("Placeholder_1:0")
    batch_size_placeholder = graph.get_tensor_by_name("Placeholder_2:0")
    attribute_or_relation = graph.get_tensor_by_name("Placeholder_3:0")
    gumbel_temp = graph.get_tensor_by_name("Placeholder_4:0")

    batch_size = num_per_image

    attribute_feed_dict = {image_feats : image_feats,  batch_size_placeholder : batch_size, attribute_or_relation : attributes_flag, gumbel_temp : 0.2}
    relation_feed_dict = {image_feats : image_feats, batch_size_placeholder : batch_size, attribute_or_relation : relations_flag, gumbel_temp : 0.2}
    attribute, a_score, a_attention_1, a_attention_2, a_attention_3 =\
        sess.run([generator_output, discriminator_output, attention_1, attention_2, attention_3], feed_dict = attribute_feed_dict)
    attribute = np.argmax(attribute, axis=2)
    a_score = np.sum(a_score, axis=1)
    relation, r_score, r_attention_1, r_attention_2, r_attention_3 =\
        sess.run([generator_output, discriminator_output, attention_1, attention_2, attention_3], feed_dict = relation_feed_dict)
    relation = np.argmax(relation, axis=2)
    r_score = np.sum(r_score, axis=1)

    a_attention_vectors = np.transpose(np.array([a_attention_1, a_attention_2, a_attention_3]), (1, 0, 2))
    r_attention_vectors = np.transpose(np.array([r_attention_1, r_attention_2, r_attention_3]), (1, 0, 2))

    attributes = [Triple(attribute[i], a_attention_vectors[i], score[i], is_attribute=True) for i in range(attribute.shape[0])]
    relations = [Triple(relation[i], r_attention_vectors[i], score[i], is_attribute=False) for i in range(relation.shape[0])]

    return attributes.extend(relations)


#Function to determine range of discriminator scores for ground truth triples
#Returns a threshold calculated as 1.5 standard deviations lower than the mean score
#assigned to a ground truth triple. This score will be used as the minimum value that
#a generated triple must achieve in order to be added to the graph
def determineThreshold(sess, graph, ground_truth_features):

    attribute_logits = []
    relation_logits = []

    discriminator_output = graph.get_tensor_by_name("Discriminator_1/transpose:0")

    real_inputs = graph.get_tensor_by_name("Placeholder:0")
    image_feats = graph.get_tensor_by_name("Placeholder_1:0")
    batch_size_placeholder = graph.get_tensor_by_name("Placeholder_2:0")
    attribute_or_relation = graph.get_tensor_by_name("Placeholder_3:0")
    gumbel_temp = graph.get_tensor_by_name("Placeholder_4:0")

    for i in range(0, ground_truth_features.shape[0], 3):
        feats = ground_truth_features[i]
        attributes = ground_truth_features[i+1]
        relations = ground_truth_features[i+2]
        attribute_feed_dict = {real_inputs : attributes, image_feats : feats, batch_size_placeholder : attributes.shape[0], attribute_or_relation : attributes_flag, gumbel_temp = 0.2}
        relation_feed_dict = {real_inputs : relations, image_feats : feats, batch_size_placeholder : relations.shape[0], attribute_or_relation : relations_flag, gumbel_temp = 0.2}
        attribute_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = attribute_feed_dict), axis=1).tolist())
        relation_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = relation_feed_dict), axis=1).tolist())

    attribute_std_dev = np.std(attribute_logits)
    attribute_mean = np.mean(attribute_logits)

    return mean - (1.5*std_dev)

#Function to filter out low score triples using the discriminator
def filterTriples(threshold, triples):
    filtered_triples = []
    for t in triples:
        if t.disc_score >= threshold:
            filtered_pairs.append(t)
    return filtered_triples

#Given two attention vectors, function to determine how similar they are
#using the generalized Jaccard similarity. The Jaccard similarity must be greater
#than or equal to a specified threshold to return True
def similarEnough(att_1, att_2, threshold = 0.75):
    intersection = sum(map(lambda x, y : min(x,y), att_1.tolist(), att_2.tolist()))
    union = sum(map(lambda x, y : max(x,y), att_1.tolist(), att_2.tolist()))

    return (intersection / union) >= threshold
        
#Function to return list of all entities in a list of triples
#Assigns a unique identifier to each entity
def findAllEntities(triples):
    all_entities = {}
    #Keep track of which triple it came from to replace duplicates later
    triple_index = 0
    for t in triples:
        if t.is_attribute:
            all_entities[t.subject] = (triple_index, 0)
        else:
            all_entities.append(t.subject)
            all_entities[t.subject] = (triple_index, 0)
            all_entities.append(t.object)
            all_entities[t.object] = (triple_index, 2)
        triple_index += 1
    return all_entities

#Function to determine the countability of an entity
#If either is uncountable (e.g snow, water, grass) then no need to perform
#entity resolution
#def determineCountability(entity):

#Function to determine all potential duplicate entity pairs
#Currently just returns all pairs, might be smarter to do something else
def determinePotentialDuplicates(all_entities_list):
    return list(itertools.combinations(all_entities_list, 2))

#Function to resolve duplicate pairs,
#TODO: Alters the triples in place
def resolveDuplicateEntities(potential_duplicates, triples):
    #For each pair of potential duplicate entities:
    ##Determine if the entity in question is countable
    ##If not, continue
    ##Otherwise, determine if they are the same entity based on the attention vector
    ##If they are the same, add both unique IDs to a list tracking the duplicate entities
    #For each entity pair in the duplicate list
    ##Add ID of pair[1] to the duplicates of pair[0]
    ##Add duplicates of pair[1] to the duplicates of pair[0]
    ##Add relations of pair[1] to the relations of pair[0]
    ##Add attributes of pair[1] to attributes of pair[0]
    ##Remove pair[1] from list of all entities
    #For each remaining entry in the list of all entities
    ##Generate pairs
    ##For each pair of entities
    ###For subject relation in pair[0]
    ####If any of pair[1]'s duplicates appears as the object 
    #####Replace the object ID with pair[1]'s unique ID
    ###For object relation in pair[0]
    ####If any of pair[1]'s duplicates appears as the subject
    #####Replace the subject ID with pair[1]'s unique ID
    ###For subject relation in pair[1]
    ####If any of pair[0]'s duplicates appears as the object 
    #####Replace the object ID with pair[0]'s unique ID
    ###For object relation in pair[1]
    ####If any of pair[0]'s duplicates appears as the subject
    #####Replace the subject ID with pair[0]'s unique ID
    #At this point the entities in the list of all entities should
    #only have relations with other entries in the list
    #Return all entities

#Function to generate a visual genome scene graph object from the entities
def generateGraph(all_entities):

#Main function
def main(path_to_model_checkpoints, batch_path):
    readVocab()
    #Read in the model
    sess, graph  = loadModel(path_to_model_checkpoints)

    #Get a list of all features for which you want to generate the triples
    filename_to_feats, filenames = getFilenameToFeatDict(batch_path)

    threshold = determineThreshold(sess, graph, ground_truth_features)

    #For each image in the filenames to feats dict
    for filename, feature in filename_to_feats.iteritems():
        #Generate triples. Each returned triple object has a subject, predicate, and object,
        #along with an attention vector and score.
        triples = generateTriples(sess, graph, feature, num_per_image = 75):
        #Filter out low probability triples via discriminator
        triples = filterTriples(threshold, triples)
        #Find all entities in the triples
        all_entities = findAllEntities(triples)
        #Determine potential duplicate entities
        potential_duplicates = determinePotentialDuplicates(list(all_entities))
        #Determine which of the entities are duplicates and save the new all_entities 
        all_entities = resolveDuplicateEntities(all_entities, potential_duplicates)
        #Generate graph from this final list of entities
        graph = generateGraph(all_entities)
        #Add graph to list of graphs
        #Generate image of scene graph next to the original image
