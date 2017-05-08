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
    <tr> <td>BPR</td><td>0.621523178808</td><td>0.331396795235</td> </tr>
    <tr> <td>RNN_Classify</td><td>0.703145695364</td><td>0.478569321339</td> </tr>
    <tr> <td>RNN_Bpr</td><td>0.715397350993</td><td>0.456694164071</td> </tr>
    <tr> <td>RNNAttention_Classify</td><td>0.702649006623</td><td>0.477714585082</td> </tr>
    <tr> <td>Attention_Classify</td><td>0.565397350993</td><td>0.376347868393</td> </tr>
    </table>
</div> 
### Neural Collaboration Filtering
<div>
    <table border="0">
    <tr> <th>Model</th><th>HR@10</th><th>NDCG@10</th> </tr>
    <tr> <td>NeuMF with Pre-training</td><td>0.730</td><td>0.447</td> </tr>
    <tr> <td>NeuMF without Pre-training</td><td>0.705</td><td>0.426</td> </tr>
    </table>
</div> 
