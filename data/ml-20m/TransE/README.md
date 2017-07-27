## Knowledge Graph 的数据
movies_id.txt --- ml-20M id : KB id
entity2id.txt --- KB id : index (对应full_entity2vec.bern里的行号)
full_*.bern   --- TransE 运行后得到的实体和关系的隐含向量表示
change_data.py --- 将上述数据处理成 {ml-20M id : vector} 的形式

## 关于relation的选择
经过数据分析，我们只选择几个特定的关系作为模型的输入
