#TODO Figure out what imports are needed for this one
import os, sys
sys.path.append(os.getcwd())

import json
import itertools
import collections
import random
import operator

import numpy as np
import tensorflow as tf

#Given two attention vectors, function to determine how similar they are
#using the generalized Jaccard similarity. The Jaccard similarity must be greater
#than or equal to a specified threshold to return True
def similarEnough(att_1, att_2, threshold = 0.85):
    intersection = np.sum(np.minimum(att_1, att_2))
    union = np.sum(np.maximum(att_1, att_2))
    #intersection = sum(map(lambda x, y : min(x,y), att_1.tolist(), att_2.tolist()))
    #union = sum(map(lambda x, y : max(x,y), att_1.tolist(), att_2.tolist()))
    
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
            all_entities[t.subject] = (triple_index, 0)
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
    count = 0
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
            #deleted_entitys_names = triples[all_entities[pair[1]][0]].triple[all_entities[pair[1]][1]].vocab_indices
            deleted_entitys_names = pair[1].vocab_indices
            for x in deleted_entitys_names:
                if x in triples[all_entities[pair[0]][0]].triple[all_entities[pair[0]][1]].vocab_indices:
                    triples[all_entities[pair[0]][0]].triple[all_entities[pair[0]][1]].vocab_indices[x] += deleted_entitys_names[x]
                else:
                    triples[all_entities[pair[0]][0]].triple[all_entities[pair[0]][1]].vocab_indices[x] = deleted_entitys_names[x]

            #Now do the actual replacement
            triples[all_entities[pair[1]][0]].triple[all_entities[pair[1]][1]] = triples[all_entities[pair[0]][0]].triple[all_entities[pair[0]][1]]
            del deleted_entitys_names
        count += 1
        del att_1
        del att_2
