import re, sys, os, numpy as np, theano
floatX = np.float32
def get_vector(filename, flag):#flag for the first info 
    file = open(filename) 
    c_data = []
    c_train = {}
    length = 0
    if flag == 1:
        length = int(re.split(" |\t|\n", file.readline())[1])
    while 1:
        line = file.readline()
        if not line:
            break
        c_data.append(line)
    for i in range(int(len(c_data))):
        line = c_data[i]
        temp_str = re.split("\t| |\n", line)
        t = []
        if temp_str[0] == "33440103" or temp_str[0] == "18261991":  # data error
            continue
	#user.append(temp_str[0])
        for j in range(1, len(temp_str)):
        #print j, temp_str[j]
            if temp_str[j] == '':
                continue
            t.append(float(temp_str[j]))
        c_train.setdefault(temp_str[0], floatX(t))
    file.close()
    length = len(c_train[c_train.keys()[0]])
    return length, c_train
def get_user(filename, flag):#flag for the first info
    file = open(filename)
    c_data = []
    c_train = []
    length = 0
    if flag == 1:
        length = int(re.split(" |\t|\n", file.readline())[1])
    while 1:
        line = file.readline()
        if not line:
            break
        c_data.append(line)
    for i in range(int(len(c_data))):
        line = c_data[i]
        temp_str = re.split("\t| |\n", line)
        t = []
        if temp_str[0] == "33440103" or temp_str[0] == "18261991":  # data error
            continue
        #user.append(temp_str[0])
	'''
        for j in range(1, len(temp_str)):
        #print j, temp_str[j]
            if temp_str[j] == '':
                continue
            t.append(float(temp_str[j]))
        c_train.setdefault(temp_str[0], floatX(t))
	'''
	c_train.append(temp_str[0])
    file.close()
    #length = len(c_train[c_train.keys()[0]])
    return c_train

def get_net(filename):
    # u -- c+ data
    file = open(filename)
    print "loading network data"
    temp_strs = re.split(" |\t|\n", file.readline())
    current_u = temp_strs[0]
    c_list = [temp_strs[1]]
    u_c = {}  #dict : u -> c+ list
    while 1:
        line = file.readline()
        #print line
        if not line:
            break
        temp_strs = re.split(" |\t|\n", line)
        s = []
        for i in temp_strs:
            if i == '':
                continue
            s.append(i)
        if(current_u != s[0]):
            u_c.setdefault(current_u, c_list)
            current_u = s[0]
            c_list = []
        c_list.append(s[1])
    file.close()      
    return u_c
def get_net_control(filename, number):
    # u -- c+ data
    file = open(filename)
    print "loading network data"
    temp_strs = re.split(" |\t|\n", file.readline())
    current_u = temp_strs[0]
    c_list = [temp_strs[1]]
    u_c = {}  #dict : u -> c+ list
    count = 0
    while 1:
        line = file.readline()
        #print line
        if not line:
            break
        temp_strs = re.split(" |\t|\n", line)
        s = []
	#count = 0
        for i in temp_strs:
            if i == '':
                continue	
	    #if count >= number:
	    #	break
	    #count += 1
            s.append(i)
        if(current_u != s[0]):
            u_c.setdefault(current_u, c_list)
            current_u = s[0]
            c_list = []
	    count = 0
	count += 1
	'''
	if count > number:
	    continue
	'''
	if float(s[2]) < 0.1:
	    continue
        c_list.append(s[1])
    file.close()
    return u_c

def get_cate_users(filename):
    # u -- c+ data
    file = open(filename)
    print "loading network data"
    cate_users = {}
    while 1:
        line = file.readline()
        #print line
        if not line:
            break
        temp_strs = re.split(" |\t|\n", line)
        s = []
        for i in temp_strs:
            if i == '':
                continue
            s.append(i)
        if int(s[1]) not in cate_users:
            cate_users[int(s[1])] = [s[0]]
        else:
            cate_users[int(s[1])].append(s[0])
        
    file.close()
    return cate_users

def get_nerual_num(line):
    str_temp = re.split(" |\r|\t|\n", line)
    nerual_num = []
    for i in str_temp:
        if i == '':
            continue
        nerual_num.append(int(i))
    return nerual_num
def read_param(params_file):
    #paramter define
    params_u = []
    params_c = []
    #read data
    f = open(params_file)
    nerual_num_u = get_nerual_num(f.readline())
    nerual_num_c = get_nerual_num(f.readline())    
    for j in range(len(nerual_num_u)-1):
        #read W
        line = f.readline()
        if not line:
            print "read data u error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
    	    temp_data.append(float(i))
        temp_data = np.array(temp_data).reshape(nerual_num_u[j], nerual_num_u[j+1])
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params_u.append(temp_data)
        #read B
        line = f.readline()
        if not line:
            print "read data u error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
            temp_data.append(float(i))
	temp_data = np.array(temp_data, dtype = np.float32)
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params_u.append(temp_data)

    for j in range(len(nerual_num_c)-1):
        #read W
        line = f.readline()
        if not line:
            print "read data c error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
    	    temp_data.append(float(i))
        temp_data = np.array(temp_data).reshape(nerual_num_c[j], nerual_num_c[j+1])
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params_c.append(temp_data)
        #read B
        line = f.readline()
        if not line:
            print "read data c error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
            temp_data.append(float(i))
	temp_data = np.array(temp_data, dtype = np.float32)
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params_c.append(temp_data)
    print "reading is finished~"
    f.close()
    return params_u, params_c, nerual_num_u, nerual_num_c
