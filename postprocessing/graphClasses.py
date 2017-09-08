import collections

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
        return collections.Counter(self.vocab_indices).most_common(1)

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
            return " ".join([self.triple[0].getName(), decoder[self.triple[1]], decoder[self.triple[2].vocab_index]])
        else:
            return " ".join([self.triple[0].getName(), decoder[self.triple[1]], self.triple[2].getName()])


