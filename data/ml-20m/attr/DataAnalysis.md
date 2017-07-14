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

## 数据分析
    从上面的属性转移矩阵和属性次数统计，可以发现，用户观看电影的原因主要受personal_appearances, actor, prequel影响，即用户看过一个电影后可能想去看这个电影中某个演员出演的电影或者这个电影的前传/先导片。（不是很明白personal_appearances 个人形象？的意义）。
    思考： 
    1、使用transE训练电影、attribute的隐含向量表示，将attribute的隐含表示作为attention的输入，还是RNN Attention Model；
    2、仿照论文《Factorizing personalized Markov chains for next-basket recommendation》，利用属性转移矩阵进行推荐，细节还没有想好。
