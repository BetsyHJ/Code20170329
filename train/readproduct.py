import re 
def readProduct(product_file):
    category = {}
    product = {}
    pro_cate = {}
    f = open(product_file)
    cate_count = 0
    while 1:
    	line = f.readline()
    	if not line:
	    break
    	temp_str = re.split(" |\n|\r|\t", line)
	pid = int(temp_str[0])
	#if pid not in pro:
	#    continue
	cate = temp_str[2]
	if cate in category:
	    cate = category[cate]
	    pro_cate.setdefault(pid, cate)
	else:
	    category.setdefault(cate, cate_count)
	    product.setdefault(cate_count, [pid])
	    pro_cate.setdefault(pid, cate_count)
	    cate_count = cate_count + 1
	    continue
	plist = product[cate]
	plist.append(pid)
	product.update({cate: plist})
    f.close()
    return category, product, pro_cate
def Product_buy(product_file, pro_sas):
    #category = {}
    #product = {}
    #pro_cate = {}
    f = open(product_file)
    #cate_count = 0
    fp = open("product_buy.txt", 'w')
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split(" |\n|\r|\t", line)
        pid = int(temp_str[0])
        #if pid not in pro:
        #    continue
        if pid in pro_sas:
	    fp.write(line)
    f.close()
    fp.close()
    #return category, product, pro_cate

def readUser_pro(file_name):
    user_pro = {}
    f = open(file_name)
    while 1:
	line = f.readline()
	if not line:
	    break
	temp_str = re.split("\r|\t| |\n", line)
	userid = temp_str[0]
	pro = []
	for i in range(1, len(temp_str)):
	    if temp_str[i] == '':
		continue
	    else:
		pro.append(temp_str[i])
	user_pro.setdefault(userid, pro)
    f.close()
    return user_pro
def readpro_nega(file_name):
    user_pro = {}
    f = open(file_name)
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split("\r|\t| |\n", line)
        userid = temp_str[0]
        pro = []
        for i in range(1, len(temp_str)):
            if temp_str[i] == '':
                continue
            else:
                pro.append(temp_str[i])
        user_pro.setdefault(userid, pro)
    f.close()
    return user_pro
def propop_order(filename):
    pro_pop = {}
    f = open(filename)
    fp = open("temp.txt", 'w')
    while 1:
	line = f.readline()
	if not line:
	    break
	temp_str = re.split("\t| |\n|\r", line)
	userid = temp_str[0]
	for i in range(1, len(temp_str)):
	    if temp_str[i] == '':
		continue
	    t = int(temp_str[i])
	    #if t not in pro:
		#continue
	    if t in pro_pop:
		pro_pop.update({t:(pro_pop[t]+1)})
	    else:
		pro_pop.setdefault(t, 1)
    #pro_pop = sorted(pro_pop.iteritems(), key=lambda d:d[1], reverse = True)
    for i in pro_pop:
        fp.write(str(i)+" "+str(pro_pop[i])+"\n")
    f.close()
    fp.close()
    return pro_pop
'''
pro_pop = propop_order()
for i in range(len(pro_pop)):
    print i, pro_pop[i]
    break
'''
'''
category, product, pro_cate = readProduct()
count = 0

#print test
for i in category:
    #if count > 10 or len(product[category[i]])<2:
    #	break
    #count = count + 1
    print i, len(product[category[i]])
print len(category)
user_pro = readUser_pro()
for i in user_pro:
    print i,user_pro[i]
    break
'''


