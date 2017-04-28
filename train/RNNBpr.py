'''
The input data is the distribution of the items.
You can use different ways to get the distribution ~
the cost funtion is bpr
'''
from keras.models import Sequential, Model
from keras.layers.core import *
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from keras.layers import Input, Add, TimeDistributed
from keras.layers import merge, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
from theano import tensor as T
from numpy import random
import re, sys, os
from read_data import get_vector
from RNNForRS import *
sample_num = 10
def bpr_batch(y_true, y_pred):
    #t = y_pred.dimshuffle(0, 2, 1)
    #print y_true.shape
    t = y_true.dimshuffle((0, 2, 1))
    scores = K.T.batched_dot(t, y_pred)
    cost = K.T.diag(scores) - scores.T
    return K.mean(-K.log(K.sigmoid(scores)))
def bpr(y_true, y_pred): #dim-2 1*d ; dim-2
    score = K.dot(y_true, y_pred.T)
    return K.mean(-K.log(K.sigmoid(T.diag(score) - score.T)), axis = -1)
def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans
def RNN_bpr(maxlen, inputDim, BPRlen = sample_num + 1, hiddenDim = 512):
    RNN_input = Input(shape=(maxlen, inputDim), dtype = 'float32', name = 'RNN_input')
    RNN_out = LSTM(hiddenDim, return_sequences = False, name = 'RNN')(RNN_input)
    RNN_out = Dropout(0.2, name = 'Dropout')(RNN_out)
    RNN_out = Dense(inputDim, activation = 'relu')(RNN_out)
    
    BPR_input = Input(shape=(BPRlen, inputDim), dtype = 'float32', name = 'BPR_input')
    
    #RNN_outs = RepeatVector(BPRlen, name = 'RNN_outs')(RNN_out)
    merged = merge([BPR_input, RNN_out], output_shape = (BPRlen,), name = 'merge1', mode = get_R)
    #merged = merge([BPR_input, RNN_outs], output_shape = (1,), name = 'merge1', mode = 'dot')
    output = Activation('softmax')(merged)
    
    #print output.get_value().shape
    model = Model(inputs = [RNN_input, BPR_input], outputs = output)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
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
def data_generator(sequences, maxlen, item_value, FeaLength):
    trainX = []
    trainY = []
    train_label = [] #for pairwise, the 1st item is 1, and the others are 0s. And the len(train_label) is sample_num
    t_label = np.zeros(sample_num + 1)
    t_label[0] = 1
    for seq in sequences: # for each sequences
        seqLength = len(seq)
	if seqLength > 200:
	    continue
        #translate the item into vector
        seq_vector = []
        for item in seq:
            seq_vector.append(item_value[item])
        #negative sample    
        samples = neg_sample(seq, item_value) # return the vectors
        #samples_size = len(samples)
        # using sliding window to get train data    
        for index in range(seqLength):
            if index + maxlen >= seqLength:
                break
            else:
                trainX.append(seq_vector[index : (index + maxlen)])
                trainY.append([seq_vector[index + maxlen]] + samples)
                train_label.append(t_label)
		#print len(seq_vector[index + maxlen]), len(samples)
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainY = np.array(trainY, dtype = 'float32').reshape(len(trainY), (sample_num + 1), FeaLength)
    train_label = np.array(train_label, dtype = 'float32')
    print trainX.shape, trainY.shape, train_label.shape
    #print "trainX is ", trainX
    return trainX, trainY, train_label
def runModelBPR(model, sequences, maxlen, items_value, FeaLength, batch_num = 2000):
    ## batch_num reset
    if batch_num < len(sequences):
        batch_num = len(sequences)
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
        trainX, trainY, train_label = data_generator(sequences_batch, maxlen, items_value, FeaLength)
        #break
        #print trainX.shape, trainY.shape, 'pp'
        # for every batch sequences, we train and update the model
        if trainX.shape[0] > 0:
            error = model.train_on_batch([trainX, trainY], train_label)

def getRNNOuput(model, layer, X_batch):
    getRNNOuput = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = getRNNOuput(X_batch) # same result as above
    return activations
            
def saveModel(getRNNOuput, sequences, maxlen, item_value, FeaLength):
    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    else:
        save_file = "userCurrentEmbedding.txt" + str(time.time())
    f = open(save_file, 'a')
    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    ValidX, _, _ = data_generator(sequences, maxlen, items_value, FeaLength)
    output = getRNNOuput([ValidX], 0)

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
    ## want to get RNN result
    #getRNNOuput = theano.function([model.layers['RNN_input'].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    getRNNOuput = K.function([model.layers['RNN_input'].input, K.learning_phase()], [model.layers['RNN_out'].output])
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
        saveModel(getRNNOuput, sequences_batch, maxlen, item_value, FeaLength)
    
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
    for epoch in range(3):
        runModelBPR(model, sequences, maxlen, items_value, FeaLength)
        #saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
	print "the iter", epoch, " is over, now it is ", time.strftime("%Y-%m-%d %X",time.localtime())
    saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)



