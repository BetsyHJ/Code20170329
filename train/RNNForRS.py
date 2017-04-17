'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
'''
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.normalization import BatchNormalization

import numpy as np
import re, sys, os, time
from read_data import get_vector

from keras import backend as K

def my_koss(y_true, y_pred):
    return K.mean(K.sigmoid(K.dot(y_true, y_pred.T)))

def RNN_for_RS(maxlen, inputDim):
    model = Sequential()
    model.add(LSTM(512, return_sequences = False, input_shape=(maxlen, inputDim)))
    model.add(Dropout(0.2))
    model.add(Dense(inputDim, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.compile(loss = 'binary_crossentropy', optimizer = 'sgd')
    #model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
    #model.compile(loss = 'squared_hinge', optimizer = 'sgd')
    model.compile(loss = my_koss, optimizer = 'sgd')

    return model

def proPrepare(filename):
    '''
        read the distribution of the items
    '''
    FeaLength, items_value = get_vector(filename, 1) #if 1, have firstline for numbers; if 0, start with data directly
    #print items_value.type
    return FeaLength, items_value #dict

def readSequences(filename):
    sequences = []
    # read lines
    f = open(filename)
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split("\r|\t| |\n", line)
        pro_list = []
        for i in range(1, len(temp_str)):
            if temp_str[i] == '':
                continue
            else:
                pro_list.append(temp_str[i])
        sequences.append(pro_list)
    f.close()
    print "the number of sequences is :", len(sequences)
    return sequences #list

def createData(sequences, maxlen, item_value, FeaLength):
    trainX = []
    trainY = []
    for seq in sequences: # for each sequences
        seqLength = len(seq)
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
                trainY.append(seq_vector[index + maxlen])
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32')#.reshape(len(trainY), 1, FeaLength)
    return trainX, trainY

def runModel(trainX, trainY, model, batch_size = 500):
    '''
    number = trainX.shape[0]
    error = []
    while 1:
	if number >= batch_size:
	    error = model.train_on_batch(trainX[number - batch_size:number], trainY[number - batch_size : number])
	    number -= batch_size
	else:
	    error = model.train_on_batch(trainX[:number], trainY[:number])
	    break
    #error /= trainX.shape[0]
    #print "error is", error
    '''
    error = model.train_on_batch(trainX, trainY)
    return error

def saveModel(model, sequences, maxlen, item_value, FeaLength):
    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    else:
        save_file = "userCurrentEmbedding.txt" + str(time.time())
    f = open(save_file, 'a')

    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    sequences_FinalMaxlen = []
    for seq in sequences:
        sequences_FinalMaxlen.append(seq[-maxlen : ] + [seq[-1]])
    validX, _ = createData(sequences_FinalMaxlen, maxlen, item_value, FeaLength)
    output = model.predict(validX, batch_size = 32)
    
    # save the output array into file
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            f.write(str(output[i][j]) + " ")
        f.write("\n")
    f.close()
    #print output

def saveModelByBatch(model, sequences, maxlen, item_value, FeaLength):
    batch_num = 100
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]    
        saveModel(model, sequences_batch, maxlen, item_value, FeaLength)


if __name__ == "__main__":

    maxlen = 5
    print "maxlen is :", maxlen
    batch_num = 500

    # read data and features
    itemFeatureFile = sys.argv[1]
    FeaLength, items_value = proPrepare(itemFeatureFile)
    sequencesFile = sys.argv[2]
    sequences = readSequences(sequencesFile)
    
    ## batch_num reset
    if batch_num < len(sequences):
        batch_num = len(sequences)
    
    # deal the shape of the input data, but because the data is big, we can not load the data in one time
    # trainX, trainY = createData(sequences, maxlen, items_value, FeaLength)

    # define the model
    inputDim = FeaLength #trainX.shape[2]
    print "the input dimension is", inputDim
    model = RNN_for_RS(maxlen, inputDim)

    # run the model
    ## model.fit(trainX, trainY, epochs = 100, batch_size = 8)
    ## the data is big, so we have to use batch to deal data
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    '''
    for iter in range(20): #for all sequence, we want set epoch 10
        for i in range(batch_num): #for the batch of the sequences
            if i == (batch_num - 1):
                sequences_batch = sequences[i*batch_size : ]
            else:
                sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
            trainX, trainY = createData(sequences_batch, maxlen, items_value, FeaLength)
            # for every batch sequences, we train and update the model
            error = runModel(trainX, trainY, model)
	print error
    # evaluate the model
    #scores = model.evaluate(trainX, trainY)
    #print scores
    '''
    #save the output of the input
    saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
