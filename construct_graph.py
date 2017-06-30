import json
import itertools
import collections
import os
import random

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
        self.og_vocab_index = vocab_index
        self.vocab_indices = [vocab_index]
        self.attributes = []
        self.duplicates = [self.unique_id]
        #self.type? 

    def addAttribute(self, attribute):
        self.attributes.append(attribute)

    def getVocabIndex(self):
        return collections.Counter(self.vocab_indices).most_common(1)

    def getName(self):
        global decoder
        index = collections.Counter(self.vocab_indices).most_common(1)
        return "{}_{}".format(decoder[index], self.unique_id)

    def decodeAttributes(self):
        return [" ".join(self.getName(), "be.v.01", decoder[att.vocab_index]) for att in self.attributes]

class Attribute(object):
    def __init__(self, vocab_index):
        self.vocab_index = vocab_index

class Triple(object):
    def __init__(self, triple, attention_vectors, disc_score, is_attribute=True):
        global entity_identifier

        self.is_attribute = is_attribute
        self.attention_vectors = attention_vectors
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
        if self.is_attribute:
            return " ".join(self.triple[0].getName(), decoder[self.triple[1]], decoder[self.triple[2]])
        else:
            return " ".join(self.triple[0].getName(), decoder[self.triple[1]], self.triple[2].getName())

"""class Graph(object):
    def __init__(self, ):
        self.nodes = []"""

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



def loadGroundTruthFeatures(batch_path):
    train_batch_path = os.path.join(batch_path, "train")

    npz1 = np.load(os.path.join(train_batch_path, "batch_0.npz"))
    npz2 = np.load(os.path.join(train_batch_path, "batch_100.npz"))
    npz3 = np.load(os.path.join(train_batch_path, "batch_300.npz"))

    big_arr_1 = npz1['arr_0']
    big_arr_2 = npz2['arr_0']
    big_arr_3 = npz3['arr_0']

    ground_truth_features = np.concatenate((big_arr_1, big_arr_2, big_arr_3))

    return ground_truth_features

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
    global vocab
    global attributes_flag
    global relations_flag

    attribute_logits = []
    relation_logits = []

    discriminator_output = graph.get_tensor_by_name("Discriminator_1/transpose:0")

    real_inputs = graph.get_tensor_by_name("Placeholder:0")
    image_feats = graph.get_tensor_by_name("Placeholder_1:0")
    batch_size_placeholder = graph.get_tensor_by_name("Placeholder_2:0")
    attribute_or_relation = graph.get_tensor_by_name("Placeholder_3:0")
    gumbel_temp = graph.get_tensor_by_name("Placeholder_4:0")

    attributes_pairs = []
    relations_pairs = []
    batch_size = 256
    for i in xrange(0, ground_truth_features.shape[0], 3):
        im_feats = ground_truth_features[i]
        attributes = ground_truth_features[i+1]
        relations = ground_truth_features[i+2]
        for a in xrange(attributes.shape[0]):
            attributes_pairs.append((im_feats, attributes[a]))
        for r in xrange(relations.shape[0]):
            relations_pairs.append((im_feats, relations[r]))

    attributes_indices = list(range(len(attributes_pairs)))
    relations_indices = list(range(len(relations_pairs)))
    random.shuffle(attributes_indices)
    random.shuffle(relations_indices)

    while len(attributes_indices) > 0:
        if len(attributes_indices) >= batch_size:
            im_batch = np.array([attributes_pairs[i][0] for i in attributes_indices[-batch_size:]], dtype=np.float32)
            triple_batch = np.array([attributes_pairs[i][1] for i in attributes_indices[-batch_size:]])
            t_batch = np.zeros((batch_size, 3, len(vocab)), dtype=np.float32)
            for row in range(t_batch.shape[0]):
                for token in range(t_batch.shape[1]):
                    t_batch[row, token, triple_batch[row, token]] = 1.0
            del attributes_indices[-batch_size:]
            attribute_feed_dict = {real_inputs : t_batch, image_feats : im_batch, batch_size_placeholder : batch_size, attribute_or_relation : attributes_flag, gumbel_temp : 0.2}
            attribute_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = attribute_feed_dict), axis=1).tolist())
        else:
            continue
            im_batch = np.array([attributes_pairs[i][0] for i in attributes_indices], dtype=np.float32)
            triple_batch = np.array([attributes_pairs[i][1] for i in attributes_indices])
            t_batch = np.zeros((len(attributes_indices), 3, len(vocab)), dtype=np.float32)
            for row in range(t_batch.shape[0]):
                for token in range(t_batch.shape[1]):
                    t_batch[row, token, triple_batch[row, token]] = 1.0
            attribute_feed_dict = {real_inputs : t_batch, image_feats : im_batch, batch_size_placeholder : len(attributes_indices), attribute_or_relation : attributes_flag, gumbel_temp : 0.2}
            attribute_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = attribute_feed_dict), axis=1).tolist())
            del attributes_indices[:]

    while len(relations_indices) > 0:
        if len(relations_indices) >= batch_size:
            im_batch = np.array([relations_pairs[i][0] for i in relations_indices[-batch_size:]], dtype=np.float32)
            triple_batch = np.array([relations_pairs[i][1] for i in relations_indices[-batch_size:]])
            t_batch = np.zeros((batch_size, 3, len(vocab)), dtype=np.float32)
            for row in range(t_batch.shape[0]):
                for token in range(t_batch.shape[1]):
                    t_batch[row, token, triple_batch[row, token]] = 1.0
            del relations_indices[-batch_size:]
            relation_feed_dict = {real_inputs : t_batch, image_feats : im_batch, batch_size_placeholder : batch_size, attribute_or_relation : relations_flag, gumbel_temp : 0.2}
            relation_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = relation_feed_dict), axis=1).tolist())
        else:
            im_batch = np.array([relations_pairs[i][0] for i in relations_indices], dtype=np.float32)
            triple_batch = np.array([relations_pairs[i][1] for i in relations_indices])
            t_batch = np.zeros((len(relations_indices), 3, len(vocab)), dtype=np.float32)
            for row in range(t_batch.shape[0]):
                for token in range(t_batch.shape[1]):
                    t_batch[row, token, triple_batch[row, token]] = 1.0
            relation_feed_dict = {real_inputs : t_batch, image_feats : im_batch, batch_size_placeholder : len(relations_indices), attribute_or_relation : relations_flag, gumbel_temp : 0.2}
            relation_logits.extend(np.sum(sess.run(discriminator_output, feed_dict = relation_feed_dict), axis=1).tolist())
            del relations_indices[:]
    
    attribute_std_dev = np.std(attribute_logits)
    attribute_mean = np.mean(attribute_logits)
    relation_std_dev = np.std(relation_logits)
    relation_mean = np.mean(relation_logits)

    return (attribute_mean - (1.5*attribute_std_dev)), (relation_mean - (1.5*relation_std_dev))

#Function to filter out low score triples using the discriminator
def filterTriples(attribute_threshold, relation_threshold, triples):
    filtered_triples = []
    for t in triples:
        if t.is_attribute:
            if t.disc_score >= attribute_threshold:
                filtered_triples.append(t)
        else:
            if t.disc_score >= relation_threshold:
                filtered_triples.append(t)
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
#Alters the triples in place
#TODO: I'm sure I should be using a better data structure here
def resolveDuplicateEntities(potential_duplicates, all_entities, triples):
    #For each pair of potential duplicate entities:
    for pair in potential_duplicates:
        #Each pair is a tuple of entity objects
        #Determine if the entity in question is countable
        #If not, continue
        #Otherwise, determine if they are the same entity based on the attention vector
        #All entities maps the entity object to a tuple. The first entry of the tuple is the index
        #of the associated triple in the triples list. The second is the index within the triple
        #of the entity (i.e whether it's a subject or object).
        att_1 = triples[all_entities[pair[0]][0]].attention_vectors[all_entities[pair[0]][1]]
        att_2 = triples[all_entities[pair[1]][0]].attention_vectors[all_entities[pair[1]][1]]
        if similarEnough(att_1, att_2):
            #If they are the same, replace entity_1 in the triples list with entity_0
            #Make sure to extend entity_0s vocab_index list with the deleted triple's vocab_index first
            deleted_entitys_names = triples[all_entities[pair[1]][0]].triple[all_entities[pair[1]][1]].vocab_indices

            triples[all_entities[pair[0]][0]].triple[all_entities[pair[0]][1]].vocab_indices.extend(deleted_entitys_names)
            
            #Now do the actual replacement
            triples[all_entities[pair[1]][0]].triple[all_entites[pair[1]][1]] = triples[all_entites[pair[0]][0]].triple[all_entities[pair[0]][1]]

def generateSamples(filename, triples, count):
        samples_dir = os.path.join("./samples", "scene_graph_samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        img = Image.open(f)
        new_im = Image.new('RGB', (224 + 300, 224))
        img = img.resize((224, 224), Image.ANTIALIAS)

        text_im = Image.new('RGB', (300, 224))
        draw = Image.Draw.Draw(text_im)
        position = 0
        for t in triples:
            s = t.decode()
            draw.text((10, 2 + (10*position)), s)
            position += 1
        new_im.paste(img, (0,0))
        new_im.paste(text_im, (224, 0))
        new_im.save(os.path.join(samples_dir, "{}.jpg".format(count)))

#Main function
def main(path_to_model_checkpoints, batch_path):
    readVocab()
    #Read in the model
    print "Loading model"
    sess, graph  = loadModel(path_to_model_checkpoints)
    print "Done"

    #Get a list of all features for which you want to generate the triples
    print "Loading filenames for sampling"
    filename_to_feats, filenames = getFilenameToFeatDict(batch_path)
    print "Done"

    print "Loading training data for threshold determination"
    ground_truth_features = loadGroundTruthFeatures(batch_path)
    print "Done"

    print "Determining threshold"
    attribute_threshold, relation_threshold = determineThreshold(sess, graph, ground_truth_features)
    print "Done"

    count = 0

    #For each image in the filenames to feats dict
    for filename, feature in filename_to_feats.iteritems():
        #Generate triples. Each returned triple object has a subject, predicate, and object,
        #along with an attention vector and score.
        triples = generateTriples(sess, graph, feature, num_per_image = 75)
        #Filter out low probability triples via discriminator
        triples = filterTriples(attribute_threshold, relation_threshold, triples)
        #Find all entities in the triples
        all_entities = findAllEntities(triples)
        #Determine potential duplicate entities
        potential_duplicates = determinePotentialDuplicates(list(all_entities))
        #Determine which of the entities are duplicates and save the new all_entities 
        resolveDuplicateEntities(potential_duplicates, all_entities, triples)
        #Generate graph from this final list of entities
        #graph = generateGraph(all_entities)
        #Add graph to list of graphs
        #Generate image of scene graph next to the original image
        generateSamples(filename, triples, count)
        count += 1


if __name__ == "__main__":
    path_to_model_checkpoints = "./logs/lstm_gen_lstm_disc_9_critic/checkpoints/"
    batch_path = "/mount/klawonnm/visual_genome/batches/"
    main(path_to_model_checkpoints, batch_path)
