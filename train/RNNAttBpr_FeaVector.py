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
def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans
def RNN_Att_bpr(maxlen, Attlen, inputDim, AttDim, outputDim, BPRlen = sample_num + 1, hiddenDim = 512):
    RNN_input = Input(shape=(maxlen, inputDim), dtype = 'float32', name = 'RNN_input')
    Att_input_start = Input(shape=(Attlen, AttDim), dtype = 'float32', name = 'Att_input_start')
    BPR_input = Input(shape=(BPRlen, inputDim), dtype = 'float32', name = 'BPR_input')

    Att_input = TimeDistributed(Dense(hiddenDim, activation = 'tanh'), name = 'Att_input')(Att_input_start)
    
    RNN_out = LSTM(hiddenDim, return_sequences = False, name = 'RNN')(RNN_input)
    RNN_out = Dropout(0.2, name = 'Dropout')(RNN_out)
    # get alpha, then calculate the Attention vector
    RNN_outs = RepeatVector(Attlen, name = 'RNN_outs')(RNN_out)
    merged = merge([Att_input, RNN_outs], name = 'merge1', mode = 'dot') 
    ##distributed = TimeDistributed(Dense(1, activation = 'tanh'), name = 'distributed1')(merged)
    ##flat_alpha = Flatten(name = "flat_alpha")(distributed)
    flat_alpha = Flatten(name = "flat_alpha")(merged)
    alpha = Dense(Attlen, activation = 'softmax', name = 'alpha')(flat_alpha)
    Att_input_trans = Permute((2, 1), name = 'Att_input_trans')(Att_input)
    r = merge([Att_input_trans, alpha], output_shape = (hiddenDim,), name = 'r_1', mode = get_R)
    #r = Reshape((hiddenDim), name = 'r1')
    ##AttDense = Dense(hiddenDim, activation = 'tanh')(r)
    # use the attention result merged with rnn out to classify

    merged = merge([r, RNN_out], mode = 'concat')
    h_ = Dense(outputDim, activation = 'tanh')(merged)
    merged = merge([BPR_input, h_], output_shape = (BPRlen,), name = 'merge2', mode = get_R)
    output = Activation('softmax')(merged)
    #print output.get_value().shape
    model = Model(inputs = [RNN_input, Att_input_start, BPR_input], outputs = output)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model#, RNN_out

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
def data_generator_withFeature(sequences, FeatureVector, maxlen, Attlen, item_value, FeaLength, AttDim):
    trainX, trainAtt, trainY = [], [], []
    train_label = [] #for pairwise, the 1st item is 1, and the others are 0s. And the len(train_label) is sample_num
    t_label = np.zeros(sample_num + 1)
    t_label[0] = 1
    for seq in sequences: # for each sequences
        #translate the item into vector
        seq_vector = []
	seq_New = []
        for item in seq:
	    if item in item_value:
                seq_vector.append(item_value[item])
		seq_New.append(item)
	seq = seq_New
	seqLength = len(seq)
        #negative sample    
        samples = neg_sample(seq, item_value) # return the vectors
        #samples_size = len(samples)
        # using sliding window to get train data    
        for index in range(seqLength):
            if index + maxlen >= seqLength:
                break
            else:
                trainX.append(seq_vector[index : (index + maxlen)])
                trainAtt.append(FeatureVector)
                trainY.append([seq_vector[index + maxlen]] + samples)
                train_label.append(t_label)
		#print len(seq_vector[index + maxlen]), len(samples)
    #print "the number of training samples is :", len(trainY)
    #print "trainX is", trainX
    #print len(trainY), maxlen, FeaLength
    trainX = np.array(trainX, dtype = 'float32').reshape(len(trainY), maxlen, FeaLength)
    trainAtt = np.array(trainAtt, dtype = 'float32').reshape(len(trainY), Attlen, AttDim)
    trainY = np.array(trainY, dtype = 'float32').reshape(len(trainY), (sample_num + 1), FeaLength)
    train_label = np.array(train_label, dtype = 'float32')
    print trainX.shape, trainAtt.shape, trainY.shape, train_label.shape
    return trainX, trainAtt, trainY, train_label
 
def runModelBPR(model, sequences, FeatureVector, maxlen, items_value, FeaLength, batch_num = 2000):
    ## batch_num reset
    if batch_num < len(sequences):
        batch_num = len(sequences)
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
        trainX, trainAtt, trainY, train_label = data_generator_withFeature(sequences_batch, FeatureVector, maxlen, Attlen, items_value, FeaLength, AttDim)
        if trainX.shape[0] > 0:
            error = model.train_on_batch([trainX, trainAtt, trainY], train_label)
	    print error
            
def saveModel(getRNNOuput, sequences, maxlen, Attlen, item_value, FeaLength, AttDim, FeatureVector):
    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    else:
        save_file = "result/userCurrentEmbedding.txt"
    f = open(save_file, 'a')
    # for every sequence we just need the final maxlen data, so we set Y as final item, default value
    sequences_Final = []
    length_control = maxlen
    #print length_control
    seq_New = []
    for seq in sequences:
	for s in seq:
	    if s in item_value:
		seq_New.append(s)
        sequences_Final.append(seq_New[-length_control : ] + [seq_New[-1]])
    #print len(sequences_Final), len(sequences_Final[0])
    #validX, _ = createData(sequences_FinalMaxlen, maxlen, item_value, FeaLength)
    #ValidX, _, _ = data_generator(sequences_FinalMaxlen, maxlen, item_value, FeaLength)
    #print ValidX.shape
    ValidX, ValidAtt, _, _ = data_generator_withFeature(sequences_Final, FeatureVector, maxlen, Attlen, item_value, FeaLength, AttDim)
    #print ValidX.shape, ValidAtt.shape
    #print len(sequences), len(sequences_Final)
    output = getRNNOuput([ValidX, ValidAtt, 0])

    # save the output array into file
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            f.write(str(output[i][j]) + " ")
        f.write("\n")
    f.close()
    #print output

def saveModelByBatch(model, sequences, maxlen, Attlen, item_value, FeaLength, AttDim, FeatureVector):
    batch_num = 100
    #sequences = sequences[:100]
    batch_size = int(round(1.0 * len(sequences) / batch_num))
    getRNNOuput = K.function([model.layers[0].input, model.layers[2].input, K.learning_phase()], model.layers[13].output)
    #getRNNOuput = K.function([model.layers[0].input, K.learning_phase()], model.layers[2].output)
    #print "get over"
    for i in range(batch_num): #for the batch of the sequences
        if i == (batch_num - 1):
            sequences_batch = sequences[i*batch_size : ]
        else:
            sequences_batch = sequences[i*batch_size : ((i + 1) * batch_size)]
        saveModel(getRNNOuput, sequences_batch, maxlen, Attlen, item_value, FeaLength, AttDim, FeatureVector)
        
def vector_string2float(svector):
    fvector = []
    svector = svector.split(" ")
    for s in svector:
        fvector.append(float(s.strip()))
    fvector = np.array(fvector)
    return fvector
    
def readFeatureVector(filename):
    f = open(filename)
    FeatureVector = []
    for line in f.readlines():
        s = line.strip().split("\t")
        FeatureVector.append(vector_string2float(" ".join(s)))
    f.close()    
    return FeatureVector
    
if __name__ == "__main__":

    maxlen = 5

    # read data and features
    itemFeatureFile = sys.argv[1]
    FeaLength, items_value = proPrepare(itemFeatureFile)
    sequencesFile = sys.argv[2]
    sequences = readSequences(sequencesFile)
    # read Knowledge Graph Feature Vector
    FeatureVector = readFeatureVector("../data/ml-20m/TransE/AttFeatureVector1.txt")
    Attlen = len(FeatureVector)
    AttDim = FeatureVector[0].shape[0]
    
    # define the model
    inputDim = FeaLength #trainX.shape[2]
    outputDim = FeaLength
    print "the input dimension is", inputDim
    model = RNN_Att_bpr(maxlen, Attlen, inputDim, AttDim, outputDim)
    print model.layers    
    #run the model
    for epoch in range(10):
        runModelBPR(model, sequences, FeatureVector, maxlen, items_value, FeaLength)
        #saveModelByBatch(model, sequences, maxlen, items_value, FeaLength)
	print "the iter", epoch, " is over, now it is ", time.strftime("%Y-%m-%d %X",time.localtime())
    saveModelByBatch(model, sequences, maxlen, Attlen, items_value, FeaLength, AttDim, FeatureVector)





