import sys;
sys.path.append("../train/")
from readproduct import readpro_nega
from read_data import get_vector
import re, os, math
import numpy as np

def readUserVector(filename, users):
    userVector = {}
    f = open(filename)
    count = 0
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split("\r|\t| |\n", line)
        t = []
        for j in temp_str:
            if j == "":
                continue
            t.append(float(j))
        userVector[users[count]] = np.float32(t)
        count += 1
    return userVector ## dict

def readUserItems(filename):
    users = []
    UserItems = {}
    # read lines
    f = open(filename)
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split("\r|\t| |\n", line)
        pro_list = []
        u = temp_str[0]
        for i in range(1, len(temp_str)):
            if temp_str[i] == '':
                continue
            else:
                pro_list.append(temp_str[i])
        UserItems[u] = pro_list
        users.append(u)
    f.close()
    return UserItems, users #dict, list

def readCandidates(filename):
    candidates = readpro_nega(filename) #test_nega
    return candidates #dict

def evaluate(trueResult, candidateResult, UsersVector, ItemsVector):
    P = 0; R = 0; MAP = 0; MRR = 0;
    # rank the candidate items
    total_pro = {}
    for i in candidateResult:
        if i not in ItemsVector:
            continue
        item = ItemsVector[i]
	#print item
	score = np.dot(item, UsersVector)
        #score = -1.0 * (np.dot(item, np.log(UsersVector)) + np.dot((1.0 - item), np.log(1.0 - UsersVector)))
	#score = np.linalg.norm(UsersVector - item)  
        total_pro[i] = score
    total_pro = sorted(total_pro.iteritems(), key=lambda d:d[1], reverse = True)
    rankedItem = []
    for (j, _) in total_pro:
        rankedItem.append(j)
    # get the evaluation
    num = [10]
    right_num = 0
    trueNum = len(trueResult)
    count = 0
    print "the rank is ", total_pro
    print "the truth is ", trueResult
    print "the candidate is ", candidateResult
    for j in rankedItem:
        if count == num[0]:
            P += 1.0 * right_num / count
            R += 1.0 * right_num / trueNum
        count += 1
        if j in trueResult:
            right_num += 1
            MAP = MAP + 1.0 * right_num / count
            if right_num == 1:
                MRR += 1.0 / count
    if right_num != 0:
        MAP /= right_num
    print P, R, MAP, MRR
    return P, R, MAP, MRR 
        

if __name__ == "__main__":
    # readUserItems
    UserItemFile = sys.argv[1]
    UserItems, users = readUserItems(UserItemFile)
    
    # read user Vector gotten from model training
    UserVectorFile = sys.argv[2] 
    FeaLength, UsersVector = get_vector(UserVectorFile, 1)
    #UsersVector = readUserVector(UserVectorFile, users)

    # read data and features
    itemFeatureFile = sys.argv[3]
    FeaLength, ItemsVector = get_vector(itemFeatureFile, 1) #if 1, have firstline for numbers; if 0, start with data directly
    #print ItemsVector[ItemsVector.keys()[0]].shape
    #readCandidates
    CandidateFile = sys.argv[4]
    candidates = readCandidates(CandidateFile)

    print "The file loading is finished, then to evalute~"

    # evaluate
    P = 0; R = 0; MAP = 0; MRR = 0;
    for u in UserItems:
        trueResult = UserItems[u]
        candidateResult = candidates[u]
        t_P, t_R, t_MAP, t_MRR = evaluate(trueResult, candidateResult, UsersVector[u], ItemsVector)
        P += t_P; R += t_R; MAP += t_MAP; MRR += t_MRR;
	#break
    number = len(UserItems)
    P /= number; R /= number; MAP /= number; MRR /= number;
    print P, R, MAP, MRR
