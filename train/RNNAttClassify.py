'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
the cost funtion is bpr
'''
from keras.models import Model
from keras.layers.core import *
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from keras.layers import Input, TimeDistributed
from keras.layers import merge, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
from theano import tensor as T
from numpy import random
import re, sys, os
from read_data import get_vector
from RNNForRS import *
Attlen, AttDis = 10, 20
def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans
def RNN_Att_Classify(maxlen, Attlen, inputDim, outputDim, hiddenDim = 512):
    RNN_input = Input(shape=(maxlen, inputDim), dtype = 'float32', name = 'RNN_input')
    Att_input_start = Input(shape=(Attlen, inputDim), dtype = 'float32', name = 'Att_input_start')
    Att_input = TimeDistributed(Dense(hiddenDim, activation = 'tanh'), name = 'Att_input')(Att_input_start)
 
    RNN_out = LSTM(hiddenDim, return_sequences = False, name = 'RNN')(RNN_input)
    RNN_out = Dropout(0.2, name = 'Dropout')(RNN_out)
    # get alpha, then calculate the Attention vector
    RNN_outs = RepeatVector(Attlen, name = 'RNN_outs')(RNN_out)
    merged = merge([Att_input, RNN_outs], name = 'merge1', mode = 'concat') 
    distributed = TimeDistributed(Dense(1, activation = 'tanh'), name = 'distributed1')(merged)
    flat_alpha = Flatten(name = "flat_alpha")(distributed)
    alpha = Dense(Attlen, activation = 'softmax', name = 'alpha')(flat_alpha)
    Att_input_trans = Permute((2, 1), name = 'Att_input_trans')(Att_input)
    r = merge([Att_input_trans, alpha], output_shape = (hiddenDim,), name = 'r_1', mode = get_R)
    #r = Reshape((hiddenDim), name = 'r1')
    ##AttDense = Dense(hiddenDim, activation = 'tanh')(r)
    # use the attention result merged with rnn out to classify

    merged = merge([r, RNN_out], mode = 'concat')
    ##merged = r
    h_ = Activation('tanh')(merged)
    output = Dense(outputDim, activation = 'softmax')(h_)
    
    #according the define before, define the model
    model = Model(input = [RNN_input, Att_input_start], output = output)
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
	if seqLength > 200:
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
                #trainY.append(one_hot(seq[(index + maxlen):], item_size)) ## m2m
		trainY.append(one_hot([seq[index + maxlen]], item_size))  ## o2o
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

def data_Att_generator(sequences, maxlen, item_value, FeaLength, items_count): #Attlen is the length which we see, and AttDis is the distance between time t (Now) and the Attention end time.
    trainX = []
    trainAtt = []
    trainY = []
    item_size = items_count # because the movie index starts from 1 
    for seq in sequences: # for each sequences
        seqLength = len(seq)
	if seqLength > 200:
	    continue
        #translate the item into vector
        seq_vector = []
        for item in seq:
            seq_vector.append(item_value[item])
        # using sliding window to get train data    
        for index in range(seqLength):
            AttStart = index + maxlen - (Attlen + AttDis)
            if index + maxlen >= seqLength:
                break
            if AttStart < 0:
		t = index + maxlen - Attlen 
		if t >= 0:
		    trainX.append(seq_vector[index : (index + maxlen)])
		    trainAtt.append(seq_vector[t : (t + Attlen)])
		    trainY.append(one_hot([seq[index + maxlen]], item_size))
                #continue
            else:
                trainX.append(seq_vector[index : (index + maxlen)])
                trainAtt.append(seq_vector[AttStart : (AttStart + Attlen)])
                #trainY.append(one_hot(seq[(index + maxlen):], item_size)) ## m2m
	        trainY.append(one_hot([seq[index + maxlen]], item_size))  ## o2o
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainAtt = np.array(trainAtt, dtype = 'float32').reshape(len(trainY), Attlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32')#.reshape(len(trainY), 1, FeaLength)
    #print "trainX is ", trainX
    return trainX, trainAtt, trainY

def data_Att_generator_save(sequences, maxlen, item_value, FeaLength, items_count): #Attlen is the length which we see, and AttDis is the distance between time t (Now) and the Attention end time.
    trainX = []
    trainAtt = []
    trainY = []
    item_size = items_count # because the movie index starts from 1
    for seq in sequences: # for each sequences
        seqLength = len(seq)
        if seqLength > 200:
            continue
        #translate the item into vector
        seq_vector = []
        for item in seq:
            seq_vector.append(item_value[item])
        # using sliding window to get train data
        index = seqLength - maxlen -1
	if index > 0:
            AttStart = index + maxlen - (Attlen + AttDis)
            if AttStart < 0:
                t = index + maxlen - Attlen
                if t >= 0:
                    trainX.append(seq_vector[index : (index + maxlen)])
                    trainAtt.append(seq_vector[t : (t + Attlen)])
                    trainY.append(one_hot([seq[index + maxlen]], item_size))
                #continue
            else:
                trainX.append(seq_vector[index : (index + maxlen)])
                trainAtt.append(seq_vector[AttStart : (AttStart + Attlen)])
                #trainY.append(one_hot(seq[(index + maxlen):], item_size)) ## m2m
                trainY.append(one_hot([seq[index + maxlen]], item_size))  ## o2o
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainAtt = np.array(trainAtt, dtype = 'float32').reshape(len(trainY), Attlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32')#.reshape(len(trainY), 1, FeaLength)
    #print "trainX is ", trainX
    return trainX, trainAtt, trainY

def read_item_fromAll(filename):
    FeaLength, ALL_value = proPrepare(itemFeatureFile)
    items_value = {}
    for item in ALL_value:
	if item[0] == 'm':
	    items_value[item] = ALL_value[item]
    return FeaLength, items_value

def saveModel(model, sequences, maxlen, item_value, FeaLength, items_count):
    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    else:
        save_file = "userCurrentEmbedding.txt" + str(time.time())
    f = open(save_file, 'a')

    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    sequences_Final = []
    length_control = max(maxlen, AttDis + Attlen)
    for seq in sequences:
        sequences_Final.append(seq[-length_control : ] + [seq[-1]])
    #validX, validAtt, _ = createData(sequences_FinalMaxlen, maxlen, item_value, FeaLength)
    validX, validAtt, _ = data_Att_generator_save(sequences_Final, maxlen, item_value, FeaLength, items_count)
    output = model.predict([validX, validAtt], batch_size = 32)

    # save the output array into file
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            f.write(str(output[i][j]) + " ")
        f.write("\n")
    f.close()
    #print output

def saveModelByBatch(model, sequences, maxlen, item_value, FeaLength, items_count):
    batch_num = 100
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
        saveModel(model, sequences_batch, maxlen, item_value, FeaLength, items_count)

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
    model = RNN_Att_Classify(maxlen, Attlen, inputDim, outputDim)
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
            #trainX, trainY = data_generator(sequences_batch, maxlen, items_value, FeaLength, items_count)
            trainX, trainAtt, trainY = data_Att_generator(sequences_batch, maxlen, items_value, FeaLength, items_count)
	    #break
	    print trainX.shape, trainAtt.shape, trainY.shape
            # for every batch sequences, we train and update the model
	    if trainX.shape[0] > 0:
    	        error = model.train_on_batch([trainX, trainAtt], trainY)
	print error
    # evaluate the model
    #scores = model.evaluate(trainX, trainY, show_accuracy=True)
    #print scores

    #save the output of the input
    saveModelByBatch(model, sequences, maxlen, items_value, FeaLength, items_count)

