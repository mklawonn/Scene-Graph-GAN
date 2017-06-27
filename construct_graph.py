import tensorflow as tf

class Entity(object):
    def __init__(self, ):
        self.unique_id = 
        #Relations in which this entity is the subject
        self.subject_relations = 
        #Relations in which this entity is the object 
        self.object_relations =
        self.attributes = 
        self.duplicates = [self.unique_id]
        self.type = 

class Graph(object):
    def __init__(self, ):
        self.nodes = []

#Function to read in the model
def

#Function to generate a bunch of triples (attributes and relations) along with the attention
#vector associated with each subject, predicate, and object
def

#Function to filter out low score triples using the discriminator
def

#Given two attention vectors, function to determine how similar they are
#For each vector, produces an ordered ranking of indices based on the size of
#the value at that index.
#Then returns True or False based on how many common indices appear in each vector's top
#ten ranking. True if that number is greater than some threshold, False otherwise
def

#Function to return list of all entities in a list of triples
#Assigns a unique identifier to each entity
def findAllEntities(triples):
    identifier = 0
    all_entities = []
    for t in triples:
        #If triple is a relation
        ##Add subject and object to all_entities
        #Else
        ##Add subject to all_entities
        identifier += 1

#Function to determine the countability of an entity
#If either is uncountable (e.g snow, water, grass) then no need to perform
#entity resolution
def determineCountability(entity):

#Function to determine all potential duplicate entity pairs based on name
def determinePotentialDuplicates(all_entities):

#Function to resolve duplicate pairs,
#returns a list of all actual entities
def resolveDuplicateEntities(all_entities, potential_duplicates):
    #Initialize list of all entities from potential duplicate entities list
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
def main(path_to_model, path_to_image_features):
#Read in the model

#Get a list of all features for which you want to generate the triples

#For each feature
    #Generate triples and attention vectors
    #Filter out low probability triples via discriminator
    #Find all entities in the triples
    all_entities = findAllEntities(triples)
    #Determine potential duplicate entities
    potential_duplicates = determinePotentialDuplicates(all_entities)
    #Determine which of the entities are duplicates and save the new all_entities 
    all_entities = resolveDuplicateEntities(all_entities, potential_duplicates)
    #Generate graph from this final list of entities
    graph = generateGraph(all_entities)
    #Add graph to list of graphs
    #Generate image of scene graph next to the original image
