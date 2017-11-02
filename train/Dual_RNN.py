'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
the cost funtion is bpr, and we make BPR and TransE jointly train
'''
from keras.models import Sequential, Model
from keras.layers.core import *
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from keras.layers import Input, Add, TimeDistributed, Embedding
from keras.layers import merge, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np, keras
from theano import tensor as T
import tensorflow as tf
from numpy import random
import re, sys, os
from read_data import get_vector
from RNNForRS import *
maxlen = 5
sample_num = 10

def get_R(X):
    Y, alpha = X[0], X[1] # pis, pu
    ans = K.batch_dot(Y, tf.expand_dims(alpha, -1))
    ans1 = tf.expand_dims(tf.matrix_diag_part(ans), -1)
    pairwise = ans1 - ans
    return pairwise
    
def add_regularizer(X):
    output1, X1, X2 = X[0], X[1], X[2] #output1, pi and pu
    ans = tf.reduce_sum(tf.abs(output1))# + 0.1 * (tf.nn.l2_loss(X1) + tf.nn.l2_loss(X2))
    return ans

def ln_sigmoid(x):
    return K.log(K.sigmoid(x))

def Dual_RNN(item_indexMax, ItemDim, BPRInputDim, TransEInputDim, hiddenDim = 128, PairwiseLen = sample_num + 1):
    # Three Input
    ItemEmbedding = Embedding(item_indexMax, ItemDim, name = "ItemEmbedding")
    BPR_RNN_Input = Input(shape=(maxlen, BPRInputDim), dtype = 'float32', name = 'BPR_RNN_input')
    TransE_RNN_Input = Input(shape=(maxlen, TransEInputDim), dtype = 'float32', name = 'TransE_RNN_input')
    ItemIds4pairwise = Input(shape=(PairwiseLen, ), dtype = 'int32', name = 'ItemIds')
    ItemEmbeddings = ItemEmbedding(ItemIds4pairwise)
    # Dual RNN
    BPR_RNN_Output = LSTM(hiddenDim, return_sequences = False, name = 'BPR_RNN', activation = 'tanh')(BPR_RNN_Input)
    TransE_RNN_Output = LSTM(hiddenDim, return_sequences = False, name = 'TransE_RNN', activation = 'tanh')(TransE_RNN_Input)
    merged = merge([BPR_RNN_Output, TransE_RNN_Output], output_shape = (2*hiddenDim,), name = 'merge', mode = 'concat')
    #merged = Flatten(name = "flat_alpha")(merged)
    Dual_RNN_Output = Dense(ItemDim, activation = 'tanh', name = "Dense")(merged)
    merged2 = merge([ItemEmbeddings, Dual_RNN_Output], output_shape = (PairwiseLen, ), name = 'merge2', mode = get_R)
    
    output1 = Activation(ln_sigmoid)(merged2)
    # regularizers
    output = merge([output1, ItemEmbeddings, Dual_RNN_Output], output_shape = (1,), name = 'output', mode = add_regularizer)
    
    model = Model(inputs = [BPR_RNN_Input, TransE_RNN_Input, ItemIds4pairwise], outputs = output)
    model.compile(loss = 'mae', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model

def neg_sample(records, indexMax):
    samples_id = []
    for i in range(sample_num):
        item_t = random.randint(0, indexMax-1)
        while (item_t in records) or (item_t in samples_id):
            item_t = random.randint(0, indexMax-1)
        samples_id.append(item_t)
    return samples_id

def data_generator(userRecords_batch, item_indexMax, transE_value, BPR_value):
    trainTransE, trainBPR, trainPairwiseId, train_label = [], [], [], []
    train_label = [] #for pairwise, the 1st item is 1, and the others are 0s. And the len(train_label) is sample_num
    t_label = np.zeros(1)
    TransEDim = transE_value[transE_value.keys()[0]].shape[0]
    BPRDim = BPR_value[BPR_value.keys()[0]].shape[0]
    for user in userRecords_batch: # for each user
	#trainX.append([user])
        records = userRecords_batch[user]
        transE_vectors, BPR_vectors, records_useful = [], [], []
        for i in records:
            if i in transE_value and i in BPR_value:
                transE_vectors.append(transE_value[i])
                BPR_vectors.append(BPR_value[i])
                records_useful.append(i)
        records = records_useful
        seqLength = len(records)
        # create windows
        for index in range(len(records)):
            if index + maxlen >= seqLength:
                break
            else:
                samples = neg_sample(records, item_indexMax)
                trainTransE.append(transE_vectors[index:(index + maxlen)])
                trainBPR.append(BPR_vectors[index:(index + maxlen)])
                trainPairwiseId.append([records[index + maxlen]] + samples) # the first one after the window
                train_label.append(t_label)
    trainTransE = np.array(trainTransE, dtype = 'float32').reshape(len(trainTransE), maxlen, TransEDim)
    trainBPR = np.array(trainBPR, dtype = 'float32').reshape(len(trainBPR), maxlen, BPRDim)
    trainPairwiseId = np.array(trainPairwiseId, dtype = 'float32')#.reshape(len(trainPairwiseId))
    train_label = np.array(train_label, dtype = 'float32')
    print trainTransE.shape, trainBPR.shape, trainPairwiseId.shape, train_label.shape
    return trainTransE, trainBPR, trainPairwiseId, train_label
    
def runModelBPR(model, userRecords_index, item_indexMax, transE_value, BPR_value, batch_num = 200):
    ## batch_num reset
    if batch_num > len(userRecords_index):
        batch_num = len(userRecords_index)
    batch_size = int(round(1.0 * len(userRecords_index) / batch_num))
    all_users = userRecords_index.keys()
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            users = all_users[i*batch_size : ]
        else:
            users = all_users[i*batch_size : ((i + 1) * batch_size)]
        userRecords_batch = {}
        for u in users:
            userRecords_batch[u] = userRecords_index[u]
        trainTransE, trainBPR, trainPairwiseId, train_label = data_generator(userRecords_batch, item_indexMax, transE_value, BPR_value)
        # for every batch sequences, we train and update the model
        if trainTransE.shape[0] > 0:
            model.train_on_batch([trainBPR, trainTransE, trainPairwiseId], train_label)

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
 	#print "output.shape is", output.shape
        for i in range(output.shape[1]):
	    #print sequences_batch
            fp.write(index2id[int(sequences_batch[i])])
            for j in range(output.shape[2]):
                fp.write(" " + str(output[0][i][j]))
            fp.write("\n")
    fp.close()
    
def saveUserEmbedding(getUserEmbedding, userRecords_index, index2user, transE_value, BPR_value, save_file):
    f = open(save_file, 'w')
    user_useful, trainTransE, trainBPR = [], [], []
    TransEDim, BPRDim = transE_value[transE_value.keys()[0]].shape[0], BPR_value[BPR_value.keys()[0]].shape[0]
    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    for u in userRecords_index:
        records = userRecords_index[u]
	#print len(records)
        transE_vectors, BPR_vectors, records_useful = [], [], []
        for i in records:
            if i in transE_value and i in BPR_value:
                transE_vectors.append(transE_value[i])
                BPR_vectors.append(BPR_value[i])
                records_useful.append(i)
        records = records_useful
        seqLength = len(records)
        # select the final maxlen data
	#print seqLength
        if len(records) < maxlen:
            continue
        user_useful.append(u)
        trainTransE.append(transE_vectors[seqLength - maxlen : seqLength])
        trainBPR.append(BPR_vectors[seqLength - maxlen : seqLength])
    trainTransE = np.array(trainTransE, dtype = 'float32').reshape(len(user_useful), maxlen, TransEDim)
    trainBPR = np.array(trainBPR, dtype = 'float32').reshape(len(user_useful), maxlen, BPRDim)    
    print trainBPR.shape, trainTransE.shape, len(user_useful)
    output = getUserEmbedding([trainTransE, trainBPR, 0])#[0]
    output = np.array(output)
    print "output.shape is", output.shape
    # save the output array into file
    for i in range(output.shape[1]):
	f.write(index2user[user_useful[i]])
        for j in range(output.shape[2]):
            f.write(" " + str(output[0][i][j]))
        f.write("\n")
    f.close()
    
def saveModelByBatch(model, userRecords_index, item_indexMax, index2item, index2user, transE_value_index, BPR_value_index):
    ## want to save user and item embeddings
    getUserEmbedding = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()], [model.layers[7].output])
    getItemEmbedding = K.function([model.layers[4].input, K.learning_phase()], [model.layers[6].output])
    item_seq = range(0, item_indexMax, 1)
    saveEmbedding(getItemEmbedding, item_seq, sample_num + 1, index2item, "temp1.txt")
    saveUserEmbedding(getUserEmbedding, userRecords_index, index2user, transE_value_index, BPR_value_index, "temp2.txt")
    
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
    return userRecords_index, index2user, index2item, item2index, user2index      
        
def name2index(transE_value, item2index):
    transE_value_index, miss_count = {}, 0
    for i in transE_value:
	if i not in item2index:
	    miss_count += 1
	    continue
        transE_value_index[item2index[i]] = transE_value[i]
    print "The TransE vector miss item", miss_count
    return transE_value_index
        
if __name__ == "__main__":
    print "read data~"
    sequencesFile = sys.argv[1]
    userRecords_index, index2user, index2item, item2index, user2index = read_userRecords(sequencesFile)
    TransEFile = sys.argv[2]
    TransEInputDim, transE_value = proPrepare(TransEFile)
    BPRFile = sys.argv[3]
    BPRInputDim, BPR_value = proPrepare(BPRFile)
    transE_value_index = name2index(transE_value, item2index)
    BPR_value_index = name2index(BPR_value, item2index)
    # define the model
    ItemDim = 128
    print "the set of item dimension is", ItemDim
    item_indexMax, user_indexMax = len(index2item), len(index2user)
    model = Dual_RNN(item_indexMax, ItemDim, BPRInputDim, TransEInputDim)
    #print "the model's layers are", model.layers    
    #for i in model.layers:
    #	print i.get_config()
    #run the model
    for epoch in range(50):
        runModelBPR(model, userRecords_index, item_indexMax, transE_value_index, BPR_value_index)
        #saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
	print "the iter", epoch, " is over, now it is ", time.strftime("%Y-%m-%d %X",time.localtime())
    saveModelByBatch(model, userRecords_index, item_indexMax, index2item, index2user, transE_value_index, BPR_value_index )


