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
	  <tr> <td>RNN</td><td>0.703145695364</td><td>0.478569321339</td> </tr>
    <tr> <td>RNNAttention</td><td>0.690231788079</td><td>0.467780340950</td> </tr>
    <tr> <td>BPR</td><td>0.621523178808</td><td>0.331396795235</td> </tr>
    </table>
</div>
