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
 
def RNN_Classify(maxlen, inputDim, outputDim):
    model = Sequential()
    model.add(LSTM(512, return_sequences = False, input_shape=(maxlen, inputDim)))
    model.add(Dropout(0.2))
    model.add(Dense(outputDim, activation='softmax'))
    #model.add(Activation('softmax'))
    #model.compile(loss = 'binary_crossentropy', optimizer = 'sgd')
    #model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
    #model.compile(loss = 'squared_hinge', optimizer = 'sgd')
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    
    return model

def one_hot(seq, item_size):
    vector = np.zeros(item_size)
    for item in seq:
        vector[int(item[1:])-1] = 1
    return vector
def data_generator(sequences, maxlen, item_value, FeaLength, items_count, sample_num = 5):
    trainX = []
    trainY = []
    item_size = items_count # because the movie index starts from 1 
    for seq in sequences: # for each sequences
        seqLength = len(seq)
	if seqLength > 1000:
	    continue
        #translate the item into vector
        seq_vector = []
        for item in seq:
            seq_vector.append(item_value[item])
        # using sliding window to get train data    
        for index in range(seqLength):
            if index + maxlen >= seqLength:
                break
            else:
                trainX.append(seq_vector[index : (index + maxlen)])
                trainY.append(one_hot(seq[(index + maxlen):], item_size))
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32')#.reshape(len(trainY), 1, FeaLength)
    #print "trainX is ", trainX
    '''
    print len(sequences), trainX.shape
    for i in range(trainX.shape[0]):
        print "trainX is", trainX[i][0]
        print "trainY is", str(trainY[i])
    '''
    return trainX, trainY
def read_item_fromAll(filename):
    FeaLength, ALL_value = proPrepare(itemFeatureFile)
    items_value = {}
    for item in ALL_value:
	if item[0] == 'm':
	    items_value[item] = ALL_value[item]
    return FeaLength, items_value

if __name__ == "__main__":

    maxlen = 5
    print "maxlen is :", maxlen
    batch_num = 2000

    # read data and features
    itemFeatureFile = sys.argv[1]
    FeaLength, items_value = read_item_fromAll(itemFeatureFile)
    sequencesFile = sys.argv[2]
    sequences = readSequences(sequencesFile)
    
    ## batch_num reset
    if batch_num < len(sequences):
        batch_num = len(sequences)
    
    # deal the shape of the input data, but because the data is big, we can not load the data in one time
    # trainX, trainY = createData(sequences, maxlen, items_value, FeaLength)

    # define the model
    items_count, outputDim = 3952, 3952
    inputDim = FeaLength #trainX.shape[2]
    print "the input dimension is", inputDim
    model = RNN_Classify(maxlen, inputDim, outputDim)
    # run the model
    ## model.fit(trainX, trainY, epochs = 100, batch_size = 8)
    ## the data is big, so we have to use batch to deal data
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    for iter in range(10): #for all sequence, we want set epoch 10
        for i in range(batch_num): #for the batch of the sequences
            if i == (batch_num - 1):
                sequences_batch = sequences[i*batch_size : ]
            else:
                sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
            trainX, trainY = data_generator(sequences_batch, maxlen, items_value, FeaLength, items_count)
	    #break
	    print trainX.shape, trainY.shape
            # for every batch sequences, we train and update the model
	    if trainX.shape[0] > 0:
    	        error = runModel(trainX, trainY, model)
	print error
    # evaluate the model
    #scores = model.evaluate(trainX, trainY, show_accuracy=True)
    #print scores

    #save the output of the input
    saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
