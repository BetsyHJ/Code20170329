import re, sys, os, random
def read_ratings(filename):
    f = open(filename)
    #UserID::MovieID::Rating::Timestamp
    user, All_ratings, item_time, item_rating= "null", {}, {}, {} #the All_ratings is {user : [[items], [scores]]}
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split(":|\n|\r|\t", line)
        #print temp_str
	ss = []
        for i in range(1, len(temp_str)):
            if temp_str[i] != "":
                ss.append(temp_str[i])
        mid, rating, timestamp = "m"+ss[0], ss[1], int(ss[2])
	if user == "null":
	    user = temp_str[0]
        if user != temp_str[0]:
            t = sorted(item_time.iteritems(), key=lambda d:d[1])
            item_time, scores, times = [], [], []#the All_ratings is {user : [[items], [scores], [times]]}
            for (item, Time) in t:
                item_time.append(item)
                scores.append(item_rating[item])
		times.append(Time)
            All_ratings[user] = [item_time, scores, times]
            # the next user
            user = temp_str[0]
            item_time, item_ranting = {}, {}
        item_time[mid] = timestamp # user : {mid : time}
        item_rating[mid] = rating # user : {mid : rating}
    # the final user
    t = sorted(item_time.iteritems(), key=lambda d:d[1])
    items, scores, times = [], [], [] #the All_ratings is {user : [[items], [scores], [times]]}
    for (item, Time) in t:
        items.append(item)
        scores.append(item_rating[item])
	times.append(Time)
    All_ratings[user] = [items, scores, times]    
    f.close()
    return All_ratings #All_ratings is {user : [[items], [scores]]}

def PrintLINE(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
        for i in range(len(items)):
            fp.write(user + "\t" + items[i] + "\t" + scores[i] + "\n")
    fp.close()

def PrintBPR(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
        for i in range(len(items)):
            fp.write(user + " " + items[i][1:] + " " + "1" + "\n")
    fp.close()

def PrintNCF(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        for i in range(len(items)-1, -1, -1):
            fp.write(user + "\t" + items[i][1:] + "\t" + str(scores[i]) + "\t" + str(times[i]) + "\n") #get items[i][1:], because the mid consist of "m" and ID
    fp.close()

def PrintSequence(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
	fp.write(user)
        for i in range(len(items)):
	    if scores[i] < 3:
		break
            fp.write(" " + items[i])
	fp.write("\n")
    fp.close()

def PrintWord2vec(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
        #fp.write(user)
        for i in range(len(items)):
            if scores[i] < 3:
                break
            fp.write(items[i] + " ")
        fp.write("\n")
    fp.close()

def generate_testNega(All_ratings, test_ratings, train_ratings, nega_num = 50):
    # get all items
    fp = open("test_nega.txt", 'w')
    All_items = []
    for user in All_ratings:
        items = All_ratings[user][0]
        for i in items:
            if i not in All_items:
                All_items.append(i)
    All_size = len(All_items)
    #user All_items to generate test_nega_sample
    for user in test_ratings:
        items, scores = test_ratings[user][0], test_ratings[user][1]
	train_items = train_ratings[user][0]
        candidates = []
	#print "item_num is", len(items)
	if len(items) * (nega_num + 1) >= All_size:
	    candidates = All_items
	    #for t_i in train_ratings[user][0]:
	    #	if t_i in candidates and t_i not in items:
	    #	    candidates.remove(t_i)
	else:
            for i in range(len(items)):
                if scores[i] < 0:  #control by score
                    break
                candidates.append(items[i])
                # generate the negative samples
                for count in range(nega_num):
                    item_t = All_items[random.randint(0, All_size - 1)]
                    while (item_t in items) or (item_t in candidates) or (item_t in train_items):
                        item_t = All_items[random.randint(0, All_size - 1)]
                    candidates.append(item_t)
	#print len(candidates)
        random.shuffle(candidates)
        fp.write(user + "\t" + " ".join(candidates) + "\n")
    fp.close()

def generate_testNega_NCF(All_ratings, test_ratings, train_ratings, nega_num = 50):
    # get all items
    fp = open("test_nega_NCF.txt", 'w')
    fp2 = open("test_nega.txt", 'w')
    All_items = []
    for user in All_ratings:
        items = All_ratings[user][0]
        for i in items:
            if i not in All_items:
                All_items.append(i)
    All_size = len(All_items)
    #user All_items to generate test_nega_sample
    for user in test_ratings:
        items, scores = test_ratings[user][0], test_ratings[user][1]
	train_items = train_ratings[user][0]
        candidates = []
	candidates2 = items[:]
        #print "item_num is", len(items)
        if len(items) * (nega_num + 1) >= All_size:
            candidates = All_items
            #for t_i in train_ratings[user][0]:
            #   if t_i in candidates and t_i not in items:
            #       candidates.remove(t_i)
        else:
            for i in range(len(items)):
                if scores[i] < 0:  #control by score
                    break
                #candidates.append(items[i])
                # generate the negative samples
                for count in range(nega_num):
                    item_t = All_items[random.randint(0, All_size - 1)]
                    while (item_t in items) or (item_t in candidates) or (item_t in train_items):
                        item_t = All_items[random.randint(0, All_size - 1)]
                    candidates.append(item_t[1:])
		    candidates2.append(item_t[:])
        #print len(candidates)
        #random.shuffle(candidates)
        fp.write("(" + user + "," + items[0][1:] + ")" + "\t" + "\t".join(candidates) + "\n")
	random.shuffle(candidates2)
	fp2.write(user + "\t" + " ".join(candidates2) + "\n")
    fp.close()

def cut_train_test_set(All_ratings, ratio): #ratio is float
    train_ratings, test_ratings = {}, {}
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        train_num = int(round(len(items) * ratio))
        train_items, train_scores, train_times = items[:train_num], scores[:train_num], times[:train_num]
        test_items, test_scores, test_times = items[train_num:], scores[train_num:], times[train_num:]
        train_ratings[user] = [train_items, train_scores, train_times]
        test_ratings[user] = [test_items, test_scores, test_times]
    #PrintLINE(train_ratings, sys.argv[1] + "train" + str(ratio) + "_LINE")
    #PrintLINE(test_ratings, sys.argv[1] + "test" + str(ratio) + "_LINE")
    #PrintSequence(test_ratings, sys.argv[1] + "_test" + str(ratio))
    #PrintBPR(train_ratings, sys.argv[1] + "train" + str(ratio) + "_BPR")
    #generate_testNega(All_ratings, test_ratings, train_ratings)
def cut_train_test_set_one_out(All_ratings): 
    train_ratings, test_ratings = {}, {}
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        train_num = len(items) - 1
        train_items, train_scores, train_times = items[:train_num], scores[:train_num], times[:train_num]
        test_items, test_scores, test_times = items[train_num:], scores[train_num:], times[train_num:]
        train_ratings[user] = [train_items, train_scores, train_times]
        test_ratings[user] = [test_items, test_scores, test_times]
    #PrintLINE(train_ratings, sys.argv[1] + "train" + str(ratio) + "_LINE")
    #PrintLINE(test_ratings, sys.argv[1] + "test" + str(ratio) + "_LINE")
    #PrintSequence(test_ratings, sys.argv[1] + "_test.oneout")
    #PrintSequence(train_ratings, sys.argv[1] + "_train.oneout")
    #PrintWord2vec(train_ratings, sys.argv[1] + "_train_w2v.oneout")
    #PrintBPR(train_ratings, sys.argv[1] + "train_BPR.oneout")
    #PrintNCF(train_ratings, sys.argv[1] + "train_NCF.oneout")
    #PrintNCF(test_ratings, sys.argv[1] + "test_NCF.oneout")
    generate_testNega_NCF(All_ratings, test_ratings, train_ratings, 99) #nega_num = 100
    #generate_testNega(All_ratings, test_ratings, train_ratings, 99) #nega_num = 100

def analysis(ratings):
    count = 0
    count_score = 0
    for user in ratings:
	items, scores = All_ratings[user][0], All_ratings[user][1]
	if len(items) > int(sys.argv[2]):
    	    count += 1
	for s in scores:
	    if int(s) == int(sys.argv[3]):
		count_score += 1
    print count, count_score
if __name__ == "__main__":
    All_ratings = read_ratings(sys.argv[1])
    #PrintNCF(All_ratings, sys.argv[1] + "train_NCF.oneout")
    #PrintLINE(All_ratings, sys.argv[1] + "_LINE")
    #cut_train_test_set(All_ratings, 0.8)
    cut_train_test_set_one_out(All_ratings)
    #analysis(All_ratings)
