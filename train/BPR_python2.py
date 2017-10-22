'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
the cost funtion is bpr without regularizer
'''
from keras.models import Sequential, Model
from keras.layers.core import *
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from keras.layers import Input, Add, TimeDistributed, Embedding
from keras.layers import merge, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

import numpy as np
from theano import tensor as T
import tensorflow as tf
from numpy import random
import re, sys, os
from read_data import get_vector
from RNNForRS import *

sample_num = 10
def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.batch_dot(Y, alpha)
    ans1 = tf.expand_dims(tf.matrix_diag_part(ans), -1)
    pairwise = ans1 - ans
    return pairwise
def ln_sigmoid(x):
    return K.log(K.sigmoid(x))
    
def bpr(item_indexMax, user_indexMax, inputDim, BPRlen = sample_num + 1):
    #TransEItemEmbedding = Input(shape=(maxlen, ), dtype = 'int32', name = 'transE_input')
    UserEmbedding = Embedding(user_indexMax, inputDim, name = "UserEmbedding")#, embeddings_regularizer = l2(1))
    ItemEmbedding = Embedding(item_indexMax, inputDim, name = "ItemEmbedding")#, embeddings_regularizer = l2(1))
    UserIds = Input(shape=(1, ), dtype = 'int32', name = 'UserIds')
    ItemIds = Input(shape=(BPRlen, ), dtype = 'int32', name = 'ItemIds')
    UserEmbeddings = UserEmbedding(UserIds)
    ItemEmbeddings = ItemEmbedding(ItemIds)
    #ItemEmbeddings = Permute((2, 1), name = 'Permute')(ItemEmbeddings)
    #merged = merge([UserEmbeddings, ItemEmbeddings], output_shape = (BPRlen,), name = 'merge', mode = get_R)
    UserEmbeddings = Permute((2, 1), name = 'Permute')(UserEmbeddings)
    merged = merge([ItemEmbeddings, UserEmbeddings], output_shape = (BPRlen,), name = 'merge', mode = get_R)
    merged = Flatten(name = "flat_alpha")(merged)
    output = Activation(ln_sigmoid)(merged)
    
    model = Model(inputs = [UserIds, ItemIds], outputs = output)
    model.compile(loss = 'mae', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model

def neg_sample(item, records, indexMax):
    samples_id = []
    for i in range(sample_num):
        item_t = random.randint(0, indexMax-1)
        while (item_t in records) or (item_t in samples_id):
            item_t = random.randint(0, indexMax-1)
        samples_id.append(item_t)
    return samples_id
    
def data_generator(userRecords_batch, item_indexMax):
    trainX, trainY = [], []
    train_label = [] #for pairwise, the 1st item is 1, and the others are 0s. And the len(train_label) is sample_num
    t_label = np.zeros(sample_num + 1)
    for user in userRecords_batch: # for each user
	#trainX.append([user])
        records = userRecords_batch[user]
        for i in records:
            samples = neg_sample(i, records, item_indexMax) #return indexs
	    trainX.append([user])
            #trainX.append((sample_num + 1) * [user])
            trainY.append([i]+samples)
            train_label.append(t_label)
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32')#.reshape(len(trainY), 1, (sample_num + 1))
    trainY = np.array(trainY, dtype = 'float32')#.reshape(len(trainY), 1, (sample_num + 1))
    train_label = np.array(train_label, dtype = 'float32')
    print trainX.shape, trainY.shape, train_label.shape
    #print "trainX is ", trainX
    return trainX, trainY, train_label
    
def runModelBPR(model, userRecords_index, item_indexMax, user_indexMax, batch_num = 200):
    ## batch_num reset
    if batch_num > len(userRecords_index):
        batch_num = len(userRecords_index)
    batch_size = int(round(1.0 * len(userRecords_index) / batch_num))
    all_users = userRecords_index.keys()
    error = []
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            users = all_users[i*batch_size : ]
        else:
            users = all_users[i*batch_size : ((i + 1) * batch_size)]
        userRecords_batch = {}
        for u in users:
            userRecords_batch[u] = userRecords_index[u]
        trainX, trainY, train_label = data_generator(userRecords_batch, item_indexMax)
        #break
        #print trainX.shape, trainY.shape, 'pp'
        # for every batch sequences, we train and update the model
        if trainX.shape[0] > 0:
            error += model.train_on_batch([trainX, trainY], train_label)
    return error 
        
def saveModel(getRNNOuput, sequences, maxlen, indexMax):
    if len(sys.argv) > 3:
        save_file = sys.argv[2]
    else:
        save_file = "userCurrentEmbedding.txt" + str(time.time())
    f = open(save_file, 'a')
    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    sequences_FinalMaxlen = []
    for seq in sequences:
	if len(seq) >= maxlen:
            sequences_FinalMaxlen.append(seq[-maxlen : ] + [seq[-1]])
	else:
	    seq = seq * maxlen * 2
	    print "seq is ", seq
	    sequences_FinalMaxlen.append(seq[-maxlen : ] + [seq[-1]])
    #validX, _ = createData(sequences_FinalMaxlen, maxlen, item_value, FeaLength)
    ValidX, _, _ = data_generator(sequences_FinalMaxlen, maxlen, indexMax)
    print ValidX.shape
    output = getRNNOuput([ValidX, 0])

    # save the output array into file
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            f.write(str(output[i][j]) + " ")
        f.write("\n")
    f.close()
    #print output

def saveEmbedding(getEmbedding, sequences, Maxlen, index2id, filename):
    fp = open(filename, 'w')
    ## batch_num reset
    batch_size = Maxlen
    batch_num = len(sequences) / Maxlen
    if len(sequences) % Maxlen != 0:
        batch_num += 1
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ] * Maxlen
            sequences_batch = sequences_batch[:Maxlen]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
            
        seq = np.array([sequences_batch], dtype = 'int32')#.reshape(len(sequences_batch), 1)
	#print "seq :", seq.shape, batch_size, Maxlen
        output = getEmbedding([seq, 0])[0]
	output = np.array(output)
 	print "output.shape is", output.shape
        for i in range(output.shape[1]):
	    #print sequences_batch
            fp.write(index2id[int(sequences_batch[i])])
            for j in range(output.shape[2]):
                fp.write(" " + str(output[0][i][j]))
            fp.write("\n")
    
    fp.close()
    
def saveModelByBatch(model, userRecords_index, item_indexMax, user_indexMax, index2item, index2user):   
    ## want to save user and item embeddings
    print  model.layers[1].get_output_shape_at(0)
    getUserEmbedding = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
    getItemEmbedding = K.function([model.layers[1].input, K.learning_phase()], [model.layers[3].output])
    item_seq, user_seq = range(0, item_indexMax, 1), range(0, user_indexMax, 1)
    saveEmbedding(getUserEmbedding, user_seq, 1, index2user, "user.BPROutput1")
    saveEmbedding(getItemEmbedding, item_seq, sample_num + 1, index2item, "item.BPROutput1")
    
def read_userRecords(filename):
    f = open(filename)
    userRecords, all_items = {}, []
    for line in f.readlines():
        s = line.strip().split()
        user = s[0]
        items = []
        for i in range(1, len(s)):
            if s[i] == '':
                continue
            items.append(s[i])
            all_items.append(s[i])
        userRecords[user] = items
    f.close()    
    print "change user and item name into index"
    # change user and item name into index
    all_items = list(set(all_items))
    all_users = userRecords.keys()
    index2user, index2item, userRecords_index = {}, {}, {}
    user2index, item2index = {}, {}
    for i in range(len(all_users)):
        index2user[i] = all_users[i]
        user2index[all_users[i]] = i
    for i in range(len(all_items)):
        index2item[i] = all_items[i]
        item2index[all_items[i]] = i
    for user in userRecords:
        userIndex = user2index[user]
        records = userRecords[user]
        itemsIndex = []
        for i in records:
            itemsIndex.append(item2index[i])
        userRecords_index[userIndex] = itemsIndex
    print "user number :", len(index2user)
    print "item number :", len(index2item)
    return userRecords_index, index2user, index2item        
        
if __name__ == "__main__":
    print "read data~"
    sequencesFile = sys.argv[1]
    userRecords_index, index2user, index2item = read_userRecords(sequencesFile)
    # define the model
    inputDim = 128
    print "the set of dimension is", inputDim
    item_indexMax, user_indexMax = len(index2item), len(index2user)
    model = bpr(item_indexMax, user_indexMax, inputDim)
    print "the model's layers are", model.layers    
    #run the model
    for epoch in range(50):
        error = runModelBPR(model, userRecords_index, item_indexMax, user_indexMax)
        #saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
	print error
	print "the iter", epoch, " is over, now it is ", time.strftime("%Y-%m-%d %X",time.localtime())
    saveModelByBatch(model, userRecords_index, item_indexMax, user_indexMax, index2item, index2user)


