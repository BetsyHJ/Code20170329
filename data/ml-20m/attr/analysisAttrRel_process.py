import re, sys, os, time, threading, multiprocessing
import pandas as pd
import numpy as np
lock = multiprocessing.Lock()
global Att_Matrix
Att_Matrix = np.zeros((27, 27))

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
    for i in range(len(atts1)):
        att1, att2, flag = atts1[i], atts2[i], 0
        for a in att1:
            if a in att2:
                flag = 1
                sim_attid.append(i)
                break
    return sim_attid    
   
def findAttNumber(rating, movie_atts, pid = ""):
    #global Att_Matrix
    Att_count = np.zeros(27)
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
                ## movie_att --p-> movie_att, count p number
	        for k in cur_sim_attid:
		    Att_count[k] += 1
    fp = open("att_count"+str(pid)+".csv", 'w')
    for i in range(Att_count.shape[0]):
        fp.write(str(int(Att_count[i])) + " ")
    #fp.write("\n")
    fp.close()

def findAttNumberInternal(rating, movie_atts, pid = "", internalNum = 5):
    #global Att_Matrix
    Att_count = np.zeros(27)
    for user in rating:
        items = rating[user]
        record_att = []
        for i in items:
            if i not in movie_atts:
                continue
            if len(record_att) < 1 + internalNum:
                cur_att = movie_atts[i]
                record_att.append(cur_att)
            else:
                cur_sim_attid = compare_atts(record_att[0], record_att[-1])
                ## movie_att --p-> movie_att, count p number
	        for k in cur_sim_attid:
		    Att_count[k] += 1
    fp = open("att_count"+str(pid)+".csv", 'w')
    for i in range(Att_count.shape[0]):
        fp.write(str(int(Att_count[i])) + " ")
    #fp.write("\n")
    fp.close()

def findFeature(rating, movie_atts, pid = "", internalNum = 0):
    #global Att_Matrix
    Path = []
    for user in rating:
        items = rating[user]
        record_att = []
        for i in items:
            if i not in movie_atts:
                continue
            if len(record_att) < 2 + internalNum:
                cur_att = movie_atts[i]
                record_att.append(cur_att)
            else:
                cur_sim_attid = compare_atts(record_att[-2], record_att[-1])
                ## movie_att --p-> movie_att, count p number
                for k in cur_sim_attid:
                    t = "u"+user + " " + "f"+k +" " + "m"+i + "\n"
		    Path.append(t)
    fp = open("UFMPath"+str(pid)+".txt", 'w')
    for i in range(len(Path)):
        fp.write(Path[i])
    #fp.write("\n")
    fp.close()

 
#Att_Matrix = np.zeros((27, 27))    
def findAttRelation(rating, movie_atts, pid = ""):
    global Att_Matrix
    Att_Matrix_temp = np.zeros((27, 27))
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
		#print pre_sim_attid, cur_sim_attid
                if pre_sim_attid == []:
                    pre_sim_attid = cur_sim_attid
                else:
                    ## if moive1 --pi-> movie2 --pj-> movie3 (pi -> pj), then mji add 1, and column normalized
                    for pi in pre_sim_attid:
                        for pj in cur_sim_attid:
                            Att_Matrix_temp[pj][pi] += 1.0
		    #print pre_sim_attid, cur_sim_attid
		    pre_sim_attid = cur_sim_attid
		pre_att = cur_att
    #with lock:
    #    Att_Matrix += Att_Matrix_temp
    saveMatrix(Att_Matrix_temp, "save"+str(pid)+".csv")
    Att_Matrix_normal = getNormalMatrix(Att_Matrix_temp)
    saveMatrix_normal(Att_Matrix_normal, "save"+str(pid)+"_normal.csv")
    '''
    data = pd.DataFrame(Att_Matrix_temp)
    data.to_csv("save"+str(pid)+".csv")
    data2 = pd.read_csv("save"+str(pid)+".csv")#.values
    print data2.index
    print data2.values
    if data == data2:
   	print "right"
    print 'wrong'
    '''
    #print Att_Matrix
    #Att_Matrix += Att_Matrix_temp
    print "chile process over~"

def findAttRelationInternal(rating, movie_atts, pid = "", internalNum = 3):
    Att_Matrix_temp = np.zeros((27, 27))
    for user in rating:
        items = rating[user]
        pre_att, pre_sim_attid = [], []
        record_att = []
        for i in items:
            if i not in movie_atts:
                continue
            if len(record_att) < 2 + internalNum:
                cur_att = movie_atts[i]
                record_att.append(cur_att)
            else:
                pre_sim_attid = compare_atts(record_att[0], record_att[internalNum])
                cur_sim_attid = compare_atts(record_att[1], record_att[internalNum+1])
                for pi in pre_sim_attid:
                    for pj in cur_sim_attid:
                        Att_Matrix_temp[pj][pi] += 1.0
                # sliding window
                cur_att = movie_atts[i]
                record_att = record_att[1:]
                record_att.append(cur_att)
    saveMatrix(Att_Matrix_temp, "save"+str(pid)+".csv")
    Att_Matrix_normal = getNormalMatrix(Att_Matrix_temp)
    saveMatrix_normal(Att_Matrix_normal, "save"+str(pid)+"_normal.csv")
    print "chile process over~"
                
def saveMatrix1(filename):
    global Att_Matrix
    l, w = Att_Matrix.shape[0], Att_Matrix.shape[1]
    fp = open(filename, 'w')
    for i in range(l):
        for j in range(w):
	    fp.write(str(int(Att_Matrix[i][j])) + " ")
	fp.write("\n")
    fp.close()
def saveMatrix(Att_Matrix, filename):
    #global Att_Matrix
    l, w = Att_Matrix.shape[0], Att_Matrix.shape[1]
    fp = open(filename, 'w')
    for i in range(l):
        for j in range(w):
            fp.write(str(int(Att_Matrix[i][j])) + " ")
        fp.write("\n")
    fp.close()

def saveMatrix_normal(Att_Matrix, filename):
    #global Att_Matrix
    l, w = Att_Matrix.shape[0], Att_Matrix.shape[1]
    fp = open(filename, 'w')
    for i in range(l):
        for j in range(w):
            fp.write(str(("%.3f" % Att_Matrix[i][j])) + " ")
        fp.write("\n")
    fp.close()
def getNormalMatrix(Matrix):
    column_sum = np.zeros(27)
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            column_sum[j] += Matrix[i][j]
    Matrix_normal = np.zeros((27, 27))
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
	    if column_sum[j] > 0:
                Matrix_normal[i][j] = 1.0 * Matrix[i][j] / column_sum[j]
	    else:
		Matrix_normal[i][j] = 0
    #saveMatrix_normal(Matrix_normal, "att_rel_normal.csv")
    #with lock:
    #    print column_sum, Matrix_normal
    return Matrix_normal

def readOneMatrix(filename):
    Matrix, count, column_sum = np.zeros((27, 27)), 0, np.zeros(27)
    f = open(filename)
    for line in f.readlines():
        numbers = line.strip().split(" ")
        for i in range(len(numbers)):
            Matrix[count][i] = int(numbers[i].strip())
	    column_sum[i] += int(numbers[i].strip())
        count += 1
    f.close()    
    return Matrix, column_sum

def readMatrix():
    Matrix = np.zeros((27, 27))
    column_sum = np.zeros(27)
    for i in range(10):
        filename = "save" + str(i) + ".csv"
        t1, t2 = readOneMatrix(filename)
  	Matrix += t1
	column_sum += t2
    saveMatrix(Matrix, "att_rel.csv")
   
    # normalize
    Matrix_normal = np.zeros((27, 27))
    for i in range(Matrix.shape[0]):
	for j in range(Matrix.shape[1]):
	    Matrix_normal[i][j] = 1.0 * Matrix[i][j] / column_sum[j]
    saveMatrix_normal(Matrix_normal, "att_rel_normal.csv")

        
if __name__ == '__main__':
    
    # deal the data by batch
    name_movielensName = read_name2movielensName("movies_id.txt")
    movie_atts = read_movie_att('direc.txt',  name_movielensName)
    rating = readSequences_normal('../ratings.csv_seq')
    rating_keys = rating.keys()#[:10]
    '''
    #global Att_Matrix
    ## threading
    thread_num = 1
    thread_dataNum = len(rating_keys) / thread_num
    #rating_keys = rating.keys()[1]
    threads = []
    #ct = CustomTask()
    for i in range(thread_num):
        t = None
        if i == thread_num - 1:
            thread_rating = {}
            for r in rating_keys[i*thread_dataNum : ]:
                thread_rating[r] = rating[r]
            #t = multiprocessing.Process(target=findAttRelationInternal, args=(thread_rating, movie_atts, i))
	    t = multiprocessing.Process(target=findAttNumberInternal, args=(thread_rating, movie_atts, i))
            t.Daemon = True
            t.start()
            
        else:
            thread_rating = {}
            for r in rating_keys[i*thread_dataNum : (i+1)*thread_dataNum]:
                thread_rating[r] = rating[r]
            #t = multiprocessing.Process(target=findAttRelationInternal, args=(thread_rating, movie_atts, i))
	    t = multiprocessing.Process(target=findAttNumberInternal, args=(thread_rating, movie_atts, i))
            t.Daemon = True
            t.start()
        threads.append(t)
    for t in threads:
        t.join()
    '''
    #print Att_Matrix
    #findAttRelation(rating, movie_atts)
    findFeature(rating, movie_atts)
    # merge the result from each batch
    #readMatrix()

