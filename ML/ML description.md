# Study of Machine-Learning

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.7/10

### Reference

- [Derivation of LSE](https://datalabbit.tistory.com/49)
- [Linear Classification](https://medium.com/elice/%EC%BB%B4%EA%B3%B5%EC%83%9D%EC%9D%98-ai-%EC%8A%A4%EC%BF%A8-%ED%95%84%EA%B8%B0-%EB%85%B8%ED%8A%B8-%E2%91%A1-%EC%84%A0%ED%98%95-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-linear-classification-model-93ba8c8fd249)
- 

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
<center>
<img src="https://i.imgur.com/WL1tIZ4.gif" width="70%" height="80%">
</center>
<br>

#### [2-1] Example code

Python code file is [hear](./ml%20algorithm/kmeans.ipynb)

- how to build KMeans model
- how to set the K value? [Elbow method / Silhouette score]

### 3. GMM