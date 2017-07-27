'''
***********  the feature info is ************
film.film.prequel       29
film.film.sequel        121
film.film.personal_appearances  141
film.actor.film 102
film.film.starring      48
media_common.adaptation.adapted_from    69
*********************************************
'''
import re, sys, os
import numpy as np
def read_KBid2ml(filename):
    f = open(filename)
    KBid2ml = {}
    for line in f.readlines():
        names = line.split(" ")
        name1, name2 = names[0], names[1].strip()[3:]
        print name1, name2
        KBid2ml[name2] = name1
    f.close()
    return KBid2ml
def read_index2KBid(filename):
    f = open(filename)
    index2KBid = {}
    for line in f.readlines():
        names = line.split("\t")
        name1, name2 = names[0].strip()[2:], int(names[1].strip())# entity id : index
        ### print name1, name2
        index2KBid[name2] = name1
	#print name2, name1
    f.close()
    return index2KBid

def readVector(filename, index2KBid, KBid2ml):
    ml_vectors = {}
    f = open(filename)
    fp = open("data.out", 'w')
    index = 0
    for line in f.readlines():
	#print index, index2KBid[index]
        if index2KBid[index] in KBid2ml:
            s = line.split("\t")
            ml_vectors[KBid2ml[index2KBid[index]]] = " ".join(s)
            fp.write("m" + KBid2ml[index2KBid[index]] + "\t" + " ".join(s))
        index += 1
    f.close()
    fp.close()
    return ml_vectors
        
def vector_string2float(svector):
    svector = svector.split(" ")
    fvector = []
    for s in svector:
        fvector.append(float(s.strip()))
    fvector = np.array(fvector)
    return fvector
    
def vector_float2string(fvector):
    m = fvector.shape[0]
    svector = []
    for i in range(m):
        svector.append(str(fvector[i]))
    svector = " ".join(svector)
    return svector            
    
def readFeatureVector(filename):
    ''' 
    index = [29, 121, 141, 102, 48, 69]
    film.actor.film 102
    film.film.starring      48
    because to find actor, movie -> starring -> actor, so we use starring_vector add actor_vector
    '''
    FeatureVector = {}
    f = open(filename)
    index = 0
    for line in f.readlines():
        s = line.strip().split("\t")
        FeatureVector[index] = " ".join(s)
        index += 1
    f.close()
    prequel_vector = FeatureVector[29]
    sequel_vector = FeatureVector[121]
    personal_appearances_vector = FeatureVector[141]
    adapted_from_vector = FeatureVector[69]
    ## deal with actor and starring
    actor_vector = vector_float2string(vector_string2float(FeatureVector[102]) + vector_string2float(FeatureVector[48]))
    fp = open("data.out2", 'w')
    fp.write(prequel_vector + "\n")
    fp.write(sequel_vector + "\n")
    fp.write(personal_appearances_vector + "\n")
    fp.write(adapted_from_vector + "\n")
    fp.write(actor_vector + "\n")
    fp.close()

if __name__ == "__main__":
    
    KBid2ml = read_KBid2ml("movies_id.txt")
    index2KBid = read_index2KBid("entity2id.txt")
    readVector("full_entity2vec.bern", index2KBid, KBid2ml)
    
    #readFeatureVector("full_relation2vec.bern")
