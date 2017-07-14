## 利用所有用户的观看记录去统计属性转移矩阵（if movie1 --attributioni-> movie2 --attributionj-> movie3, then mji add 1）
其中，矩阵按列做过归一化处理，橙色表示转移概率>=0.1的；蓝色表示转移概率介于0.09-0.1的；紫色表示转移概率介于0.08-0.09的。
![image](https://github.com/BetsyHJ/Code20170329/blob/develop/data/ml-20m/attr/Un-personalAttMC.JPG)
## 选取三个用户，统计单用户的属性转移矩阵
### user0
![image](https://github.com/BetsyHJ/Code20170329/blob/develop/data/ml-20m/attr/User0AttMC.JPG)
### user1
![image](https://github.com/BetsyHJ/Code20170329/blob/develop/data/ml-20m/attr/User1AttMC.JPG)
### user2
![image](https://github.com/BetsyHJ/Code20170329/blob/develop/data/ml-20m/attr/User2AttMC.JPG)

## 统计属性的次数（if movie1 --attributei-> movie2, then vi add 1）
实际上，电影的语言、国家这样的属性太过宽泛，信息量不够；而actor、prequel这类属性的更有意义。
![image](https://github.com/BetsyHJ/Code20170329/blob/develop/data/ml-20m/attr/att_count.JPG)