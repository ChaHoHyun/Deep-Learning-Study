# Study of Machine-Learning

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.7/10

### Reference

- [Derivation of LSE](https://datalabbit.tistory.com/49)
- [Linear Classification](https://medium.com/elice/%EC%BB%B4%EA%B3%B5%EC%83%9D%EC%9D%98-ai-%EC%8A%A4%EC%BF%A8-%ED%95%84%EA%B8%B0-%EB%85%B8%ED%8A%B8-%E2%91%A1-%EC%84%A0%ED%98%95-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-linear-classification-model-93ba8c8fd249)
- [Gaussian Mixture Model](https://angeloyeo.github.io/2021/02/08/GMM_and_EM.html)

### Index

1. [Linear Regression](#1-linear-regression)
2. [KMeans](#2-kmeanskmm-k-means-clustering)
3. [GMM](#3-gmm-gaussian-mixture-model)
4. [PCA / LDA](#4-pca--lda)

### 1. Linear Regression
<br>

 Linear regression is perhaps one of the most well known and well understood algorithms in statistics and machine learning. It is one of the Supervised Learning and regression problem since the value that i want to estimate is the real value.

- A type of Supervised Learning
- Linear Regression is based on **LSE**(Least Square Error), which can minimize loss.

 1-1) Concept : If we can assume that there is a <u>**linear relationship**</u> between X (input) and y (Lable), we can find the weight and bias of the **linear** regression relationship. ↔ **Logistic Regression**

 1-2) Hypothesis : $H_(x) = Wx + b$

 1-3) Cost Function : LSE [here](#1-1-proof-of-linear-regression)

<br>
<center><img src="https://www.reneshbedre.com/assets/posts/reg/reg_front.svg" width="70%" height="100%"></center>

<center>[Figure of Linear Regression]</center>

#### [1-1] Proof of Linear Regression

We can derive Linear Regressiom as below.<br>
[Reference - Proof of Linear Regerssion](https://datalabbit.tistory.com/49)

$y_{i} = \beta_{0} + \beta_{1}x_{i}+\epsilon_{i}$<br>
$\epsilon_{i} = y_{i}-\beta_{0} - \beta_{1}x_{i}$
 > LSE → $MinS^2 = Min \sum \limits_{i=1}^{n} \epsilon_{i}^2 = Min \sum \limits_{i=1}^{n} (y_{i}-\beta_{0} - \beta_{1}x_{i})^2$

 - condition<br>
 
    1) $f^\prime(x) = 0$ 
       -  $\beta_{1} = \frac{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})^2} = \frac{Cov(X,Y)}{Var(X)}$
       -  $\beta_{0} = \bar{y} - \beta_{1}\bar{x}$
       
       <br>
    2) $f''(x) > 0$ => $Satisfied$

    <center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbmEJi6%2FbtqIX4TtDci%2FvkuSWkUky1OvSgdYymDtmK%2Fimg.png" width="50%" height="100%"></center>
<center>[Minimizing the Error function]</center>
<br>

#### [1-2] Example code

Python code file is [hear](./ml%20algorithm/linear_regression.ipynb)


### 2. Kmeans(KMM, K-means clustering)

 The goal of this algorithm is to find groups in the data, with the number of groups represented by the **variable K**. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

- A type of Unsupervised Learning (Non-Label)

 1-1) Concept : If we can assume that there is a <u>**linear relationship**</u> between X (input) and y (Lable), we can find the weight and bias of the **linear** regression relationship. ↔ **Logistic Regression**

 1-2) Example<br>
     - Recommendation Engine: Tie up similar products to personalize the user experience <br>
     - Search engine: Tie related topics or search results <br>
     - Segmentation: Tie up similar customers according to region, demographics, and behavior <br>

 1-3) Theory 
1. Decide K arbitrary center points(centroids)
2. Devide each data in the group to which the nearest centroids belong
3. Update the center point of each cluster based on the data belonging to each cluster 
4. Repeat steps 2. and 3. until the center point is no longer updated

<br>
<center><img src="https://i.imgur.com/WL1tIZ4.gif" width="70%" height="80%"></center>
<center>[Process of KMeans Clustering]</center>
<br>

#### [2-1] Example code

Python code file is [hear](./ml%20algorithm/kmeans.ipynb)

- how to build KMeans model
- how to set the K value? [Elbow method / Silhouette score]

#### [2-2] What is EM?

The EM algorithm is basically an algorithm mainly used for unsupervised learning. 
The EM algorithm can be divided into two stages: 1) E-step and 2) M-step.
In conclusion, it is a method of finding the optimal parameter value by repeating E-step and M-step.

 - E-step: Calculate the Likelihood value as close as possible to Likelihood from the initial value of any given parameter.
 - M-step : Obtain a new parameter value that maximizes the likelihood calculated in E-step.
 - Repeat the above two steps continuously until the parameter value does not changed.
   
   ※ MLE(Maximum Likelihood Estimation)?<br>
     To estimate the maximizing probability p(x) in a probability density function of an event we do not know.<br>
     > Ex) What is the probability that if a coin is thrown 1,000 times and the front is 600 times? <br> → We'll say 0.6, which we can say on the based on the MLE.

<br>

### 3. GMM (Gaussian Mixture Model)

To cluster data under the assumption that it is a mixture of data sets with multiple Gaussian distributions.

- A type of Unsupervised Learning
- What is Gaussian distributions? (Gaussian = Normal distribution) : <br>
  The distribution of the data is symmetrically represented by the mean value

<center><img src="https://www.researchgate.net/profile/Jan-Bender-2/publication/334535945/figure/fig1/AS:781913300144134@1563434066572/Gaussian-bell-function-normal-distribution-N-0-s-2-with-varying-variance-s-2-For.png" width="60%" height="80%"></center>
<center>[Gaussian/Normal Distribution]</center>
<br>

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fmwjs7%2FbtqJWr0ql2d%2FOno3SQz9nkviLPIqDbCNd0%2Fimg.png" width="70%" height="80%"></center>
<center>[Gaussian Mixture Model]</center>

 - Different with KMeans : GMM is probability-based clustering and K-Means is distance-based clustering. Therefore, K-Means is a method of clustering while moving the center on a distance-based, which is more effective when the data within an individual cluster are distributed into a circle.(↔ GMM : Ellipse)

<br>
<center>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkSQUA%2FbtqGCKvhPVL%2FK92cpZ4pKKf5nFyvc8dXg0%2Fimg.png" width="40%" height="80%">
</center>
<center>[GMM vs KMeans]</center>

[Reference] [News on the Development of Kakao Speaking Recognition Using GMM](https://papago.naver.com/)

#### [3-1] Theory

<br>
<center>
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2021-02-08-GMM_and_EM/pic6.png" width="80%" height="80%">
</center>
<center>
1) Set randomly number of 'n_component' Gaussian distributions when we do not know abount the label. Through two given Gaussian distribution above figure, it is possible to decide Label for all data samples.
</center>

<br>
<center>
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2021-02-08-GMM_and_EM/pic7.png" width="70%" height="80%">
</center>
<center>
2) The above results can be obtained through process of 1).
</center>

<br>
<center>
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2021-02-08-GMM_and_EM/pic8.png" width="70%" height="80%">
</center>
<center>
3) If the Gaussian distribution is drawn again as a result 2) labeling , it will be like above.
</center>

<br>
<center>
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2021-02-08-GMM_and_EM/pic9.png" width="70%" height="80%">
</center>
<center>4) Repeat n times until not changing.</center>

<br>
<center><img src="https://aabkn.github.io/assets/density_estimation.gif" width="70%" height="80%"></center>

<br>

#### [3-2] Example code

Python code file is [hear](./ml%20algorithm/GMM.ipynb)

- KMeans vs GMM

<br>


### 4. PCA & LDA

<br>
1) PCA(Principal Component Analysis) : How to find the most similar lower dimension data from higher dimension data(=dimension reduction)
 → Therefore, PCA is mainly used to model with new variables by combining existing variables when there are too many variables.

- A type of Unsupervised Learning

<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0.fpng/?fname=http://t1.daumcdn.net/brunch/service/user/b34P/image/r-D5AEAOuZM_ZZPQZKPo6GRBilk.png" width="70%" height="80%"></center>
<center>[Description of PCA]</center>