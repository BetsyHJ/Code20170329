import re, sys, os
import numpy as np

def read_name2movielensName(filename):
    f = open(filename)
    name_movielensName = {}
    for line in f.readlines():
        names = line.split(" ")
        name1, name2 = names[0], names[1].strip()[3:]
        ### print name1, name2
        name_movielensName[name2] = name1
    f.close()
    return name_movielensName
    
def read_movie_att(filename, name_movielensName):
    f = open(filename)
    movie_info = {} #{movieid:[att1s, att2s, ... , att27s]}
    for line in f.readlines():
        ## for each movie
        info = line.split("\t")
        name = 'm' + name_movielensName[info[0][2:]]
        movie_atts = []
        ## for each attribute
        for i in range(1, len(info)):
            entities = info[i].split(":")
            t = []
            if len(entities) > 1:
                entities = entities[1].split(",")
                for e in range(len(entities)):
                    t.append(entities[e].strip())
            movie_atts.append(t)
        movie_info[name] = movie_atts
        # print len(movie_info[name]), name, movie_atts
    f.close()
    print len(movie_info)
    return movie_info
    
def readSequences_normal(filename):
    ratings = {} #{user:[item]}
    # read lines
    f = open(filename)
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split("\r|\t| |\n", line)
        s = []
        for i in temp_str:
            if i != '':
                s.append(i)
        pro_list = []
        for i in range(1, len(s)):
            pro_list.append(s[i])
        ratings[s[0]] = pro_list
    f.close()
    print "the number of sequences is :", len(ratings)
    return ratings #list

def compare_atts(atts1, atts2):
    sim_attid = [] ## return the id of the same attribute between atts1 and atts2
    for i in range(atts1):
        att1, att2, flag = atts1[i], atts2[i], 0
        for a in att1:
            if a in att2:
                flag = 1
                sim_attid.append(i)
                break
    return sim_attid    
    
def findAttRelation(rating, movie_atts):
    Att_Matrix = np.zeros((27, 27), dtype = "float32")
    for user in rating:
        items = rating[user]
        pre_att, pre_sim_attid = [], []
        for i in items:
            if i not in movie_atts:
                continue
            if len(pre_att) == 0:
                pre_att = movie_atts[i]
            else:
                cur_att = movie_atts[i]
                cur_sim_attid = compare_atts(pre_att, cur_att)
                if pre_sim_attid == []:
                    pre_sim_attid = cur_sim_attid
                else:
                    ## if moive1 --pi-> movie2 --pj-> movie3 (pi -> pj), then mji add 1, and column normalized
                    for pi in pre_sim_attid:
                        for pj in cur_sim_attid:
                            Att_Matrix[pj][pi] += 1.0
    print Att_Matrix
                
        
if __name__ == '__main__':
    name_movielensName = read_name2movielensName("movies_id.txt")
    movie_atts = read_movie_att('direc.txt',  name_movielensName)
    rating = readSequences_normal('ratings.csv_seq')
    findAttRelation(rating, movie_atts)