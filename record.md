## Add fixed user vector
<div>
    <table border="0">
    <tr> <th>Model</th><th>P@10</th><th>R@10</th><th>MAP</th><th>MRR</th><th>HR@10</th><th>NDCG@10</th> </tr>
    <tr> <td>non-UserVec</td><td>0.276</td><td>0.244</td><td>0.273</td><td>0.515</td><td>0.853</td><td>0.591</td> </tr>
    <tr> <td>add UserVec</td><td>0.261</td><td>0.234</td><td>0.260</td><td>0.500</td><td>0.829</td><td>0.570</td> </tr>
    </table>
</div>

## JD Dataset Test
由于jd数据中item数量16w+，因此将loss function由classification换成pairwise

## 模型变形
训练时使用Atten，测试时不使用

## 数据分析
https://github.com/BetsyHJ/Code20170329/blob/develop/Readme
