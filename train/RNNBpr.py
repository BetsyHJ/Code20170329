'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
the cost funtion is bpr
'''
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
from theano import tensor as T
from numpy import random
import re, sys, os
from read_data import get_vector
from RNNForRS import *

def bpr_batch(y_true, y_pred):
    scores = K.dot(y_pred, y_true.T)
    scores = scores.reshape(scores.shape[0], scores.shape[-1])
    t = np.zeros((scores.shape[-1], 1)); t[0][0] = 1;
    cost = K.dot(scores, t) - scores
    return K.cast(K.mean(-K.log(K.sigmoid(scores))), float)
def bpr(y_true, y_pred): #dim-2 1*d ; dim-2
    score = T.diag(K.dot(y_true, y_pred.T))
    return K.mean(-K.log(K.sigmoid(T.diag(score) - score.T)), axis = -1)
    
def RNN_bpr(maxlen, inputDim):
    model = Sequential()
    model.add(LSTM(512, return_sequences = False, input_shape=(maxlen, inputDim)))
    model.add(Dropout(0.2))
    model.add(Dense(inputDim, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.compile(loss = 'binary_crossentropy', optimizer = 'sgd')
    #model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
    #model.compile(loss = 'squared_hinge', optimizer = 'sgd')
    model.compile(loss = bpr, optimizer = 'sgd')

    return model

def neg_sample(seq, items_value, sample_num = 10):
    All_items = items_value.keys()
    All_size = len(All_items)
    samples_id, samples = [], []
    for i in range(sample_num):
        item_t = All_items[random.randint(0, All_size-1)]
        while (item_t in seq) or (item_t in samples_id):
            item_t = All_items[random.randint(0, All_size-1)]
        samples_id.append(item_t)
        samples.append(items_value[item_t])
    return samples

def runModelBPR(model, sequences, maxlen, item_value, FeaLength, sample_num = 5):
    for seq in sequences: # for each sequences
        seqLength = len(seq)
        #translate the item into vector
        seq_vector = []
        for item in seq:
            seq_vector.append(item_value[item])
        # using sliding window to get train data   
        error = []
	samples = neg_sample(seq, item_value) # return the vectors
	samples_size = len(samples)
        for index in range(seqLength):
            if index + maxlen >= seqLength:
                break
            else:
                #samples = neg_sample(seq[index + maxlen :], item_value) # return the vectors
                trainX = [seq_vector[index : (index + maxlen)] for copy in range(1 + samples_size)]
                trainY = [seq_vector[index + maxlen]] + samples
                trainX = np.array(trainX, dtype = 'float32').reshape(1 + samples_size, maxlen, FeaLength)
                trainY = np.array(trainY, dtype = 'float32')
		er = runModel(trainX, trainY, model)
		#print er
		error.append(er)
                #error.append(runModel(trainX, trainY, model))
        print "the error is ", np.mean(error)

if __name__ == "__main__":

    maxlen = 5

    # read data and features
    itemFeatureFile = sys.argv[1]
    FeaLength, items_value = proPrepare(itemFeatureFile)
    sequencesFile = sys.argv[2]
    sequences = readSequences(sequencesFile)

    # define the model
    inputDim = FeaLength #trainX.shape[2]
    print "the input dimension is", inputDim
    model = RNN_bpr(maxlen, inputDim)
    
    #run the model
    for epoch in range(10):
        runModelBPR(model, sequences, maxlen, items_value, FeaLength)
        saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
	print "the iter", epoch, " is over, now it is ", time.strftime("%Y-%m-%d %X",time.localtime())


