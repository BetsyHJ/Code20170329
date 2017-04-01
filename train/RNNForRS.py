'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
'''
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import re, sys, os
from read_data import get_vector
def RNN_for_RS(maxlen, inputDim):
    model = Sequential()
    model.add(LSTM(512, return_sequences = False, input_shape=(maxlen, inputDim)))
    model.add(Dropout(0.2))
    model.add(Dense(inputDim))
    #model.add(Activation('softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'sgd')

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
    print "the number of training samples is :", len(trainY)
    trainX = np.array(trainX, dtype = 'float32')#.reshape(, maxlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32')#.reshape(, maxlen, FeaLength)
    return trainX, trainY

if __name__ == "__main__":

    maxlen = 2
    print "maxlen is :", maxlen
    # read data and features
    itemFeatureFile = sys.argv[1]
    FeaLength, items_value = proPrepare(itemFeatureFile)
    sequencesFile = sys.argv[2]
    sequences = readSequences(sequencesFile)
        
    # deal the shape of the input data
    trainX, trainY = createData(sequences, maxlen, items_value, FeaLength)

    # define the model
    inputDim = trainX.shape[2]
    print "the input dimension is", inputDim
    model = RNN_for_RS(maxlen, inputDim)

    # run the model
    model.fit(trainX, trainY, epochs = 100, batch_size = 8)
    # evaluate the model
    scores = model.evaluate(trainX, trainY)
    print scores

