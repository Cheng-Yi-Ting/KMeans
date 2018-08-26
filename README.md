# k-平均演算法(k-means clustering)
此專案以Python3進行開發，使用scikit-learn以新聞資料進行TF-IDF，結合K-Means分群實作的範例。

### K-means Introduction:
```
K-means 是經典的分群演算法，目標是分成 k 個不同的群。方法步驟如下：
step1. 隨機任挑選 k 個點作為中心點，分為 k 群。
step2. 每一點計算與中心點的距離，判斷該點是哪一群。
step3. 每一群內重新計算平均值，作為新的中心點。
step4. 回到第二步，重新分群，直到分群結果固定。
```
![image](https://github.com/Cheng-Yi-Ting/KMeans/blob/master/img/center points & clustering result.png)
![image](https://github.com/Larix/TF-IDF_Tutorial/blob/master/img/result.png)


### 補充:
新聞資料只有550篇，斷詞使用jieba，k-means需要先設定k值(分群數量)。可以看到分群準確度並不太好，未來可考慮不同的分群法，如階層式分群法。
