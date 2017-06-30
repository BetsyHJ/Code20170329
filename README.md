# Code20170329
## install
install keras https://github.com/ogrisel/keras
## run
```
cd train
python RNNForRS.py ../data/testModelFeature.txt ../data/testModelRecord2.py
```
You can also use command to set the output file
```
python RNNForRS.py ../data/testModelFeature.txt ../data/testModelRecord2.py Output.txt
```
## result
### movielens1M
<div>
    <table border="0">
    <tr> <th>Model</th><th>HR@10</th><th>NDCG@10</th> </tr>
    <tr> <td>BPR</td><td>0.718</td><td>0.441</td> </tr>
    <tr> <td>RNN_Classify</td><td>0.730</td><td>0.524</td> </tr>
    <tr> <td>RNN_Bpr</td><td>0.751</td><td>0.505</td> </tr>
    <tr> <td>RNNAttention_Classify</td><td>0.731</td><td>0.525</td> </tr>
    <tr> <td>Attention_Classify</td><td>0.565</td><td>0.376</td> </tr>
    </table>
</div> 
<div>
    <table border="0">
    <tr> <th>Model</th><th>HR@10</th><th>NDCG@10</th> </tr>
    <tr> <td>NeuMF with Pre-training</td><td>0.730</td><td>0.447</td> </tr>
    <tr> <td>NeuMF without Pre-training</td><td>0.705</td><td>0.426</td> </tr>
    </table>
</div> 
