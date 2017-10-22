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
        name1, name2 = names[0], "m." + names[1].strip()[3:]
        #print name1, name2
        KBid2ml[name2] = name1
    f.close()
    return KBid2ml
def read_index2KBid(filename):
    f = open(filename)
    index2KBid = {}
    for line in f.readlines():
        names = line.split("\t")
        name1, name2 = names[0].strip(), int(names[1].strip())# entity id : index
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
       
def readVector2(filename, index2KBid):
    ml_vectors = {}
    f = open(filename)
    fp = open("data.out", 'w')
    index = 0
    for line in f.readlines():
        #print index, index2KBid[index]
        if True:
            s = line.split("\t")
            ml_vectors[index2KBid[index]] = " ".join(s)
            fp.write(index2KBid[index] + "\t" + " ".join(s))
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
   
def readFeature2(filename):
    f = open(filename)
    index =[190, 29, 196, 850, 83, 1118, 1102, 1287, 170, 1104, 724, 714, 869, 775, 279, 734, 546, 1091, 1196, 768, 691, 773, 735, 99, 68, 692]
    lines = f.readlines()
    # to get actor vector
    t1 = " ".join(lines[574].strip().split("\t"))
    t2 = " ".join(lines[325].strip().split("\t"))
    actor_vector = vector_float2string(vector_string2float(t1) + vector_string2float(t2))
    features = []
    for i in index:
	s = lines[i].strip().split("\t")
	features.append(" ".join(s))
    f.close()
    fp = open("data.out", 'w')
    fp.write("0\t"+actor_vector+"\n")
    for i in range(len(features)):
	fp.write(str(i+1) + "\t" + features[i]+ "\n")
    fp.close()
	
 
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
    prequel_vector = FeatureVector[30]
    sequel_vector = FeatureVector[191]
    personal_appearances_vector = FeatureVector[776]
    adapted_from_vector = FeatureVector[126]
    ## deal with actor and starring
    actor_vector = vector_float2string(vector_string2float(FeatureVector[574]) + vector_string2float(FeatureVector[325]))
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
    #readVector("entity2vec.vec", index2KBid, KBid2ml)
    #readVector2("entity2vec.vec", index2KBid)

    #readFeature2("relation2vec.vec")
    readFeatureVector("relation2vec.vec")
