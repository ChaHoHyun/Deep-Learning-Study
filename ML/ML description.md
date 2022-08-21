# Study of Machine-Learning

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.8/8

### Reference

- [Scikit-learn Official Document](https://scikit-learn.org/stable/)
- [Derivation of LSE](https://datalabbit.tistory.com/49)
- [Linear Classification](https://medium.com/elice/%EC%BB%B4%EA%B3%B5%EC%83%9D%EC%9D%98-ai-%EC%8A%A4%EC%BF%A8-%ED%95%84%EA%B8%B0-%EB%85%B8%ED%8A%B8-%E2%91%A1-%EC%84%A0%ED%98%95-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-linear-classification-model-93ba8c8fd249)
- [Gaussian Mixture Model](https://angeloyeo.github.io/2021/02/08/GMM_and_EM.html)
- [Data Scaling](https://dacon.io/codeshare/4526)
- [K-Nearest Neighbor](https://dacon.io/codeshare/4526)
- [What is Ensemble Learning](https://www.projectpro.io/article/a-comprehensive-guide-to-ensemble-learning-methods/432#toc-3)
- [Boosting - GBM & XGBoost & LightGBM](https://velog.io/@dbj2000/ML)
- [kaggle Titanic Example - Sungwookle](http://sungwookle.site/research/2106211010/)

### Index

1. [Linear Regression](#1-linear-regression)
2. [KMeans](#2-kmeanskmm-k-means-clustering)
3. [GMM](#3-gmm-gaussian-mixture-model)
4. [PCA / LDA](#4-pca--lda)<br>
   4-3. [Scaler](#4-3-scaler)
5. [KNN](#5-knn-k-nearest-neighbors)<br>
   5-1. [Cross Validation](#5-1-cross-validation)<br>
   5-2. [Grid Search](#5-2-grid-search)
7. [DecisionTreeClassifier](#7-decisiontreeclassifier)<br>
   7-1. [Ensemble Learning](#7-1-ensemble-learning)<br>
   7-2. [Bagging Classifier](#7-2-bagging-classifier)<br>
   7-2. [1] [Random Forest](#7-2-1-random-forest)<br>
   7-3. [Voting Classifier](#7-3-voting-classifier)<br>
   7-4. [Boosting](#7-4-boosting)
   

<br>

### Choosing Model

<br>

<center><img src="https://scikit-learn.org/stable/_static/ml_map.png" width="90%" height="100%"></center>

[[reference] Choosing the right estimator - Sklearn](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
<br><br>

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

 1-1) Concept : Clustering according to the number(k) of each data group

  ※ <u>**Scaler Required**</u>

<br>
<center><img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png" width="100%" height="80%"></center>
<center>[Example of Clustering]</center>
<br>

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

#### 4-1. PCA

<br>
1) PCA(Principal Component Analysis) : How to find the most similar lower dimension data from higher dimension data(=dimension reduction)
 → Therefore, PCA is mainly used to model with new variables by combining existing variables when there are too many dimensions.<br>

- Dimension reduction techniques applied in Unsupervised Learning

<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0.fpng/?fname=http://t1.daumcdn.net/brunch/service/user/b34P/image/r-D5AEAOuZM_ZZPQZKPo6GRBilk.png" width="70%" height="80%"></center>
<center>[Description of PCA]</center>

  1-1) Why reduce dimensions(PCA)?<br>
  -  Feature Selection : It is determined through the correlation coefficient value, and unnecessary features are discarded.
  - Feature Extraction 
  - Feature Generation : New features are created from the data and the purpose of machine learning algorithm.

  1-2) Caution
   - Since PCA uses a covariance matrix, it requires a process of **Scaler** each feature data with different units and distributions.
   - n_components == min(n_samples, n_features)
   - Usually, the accuracy is less than before. because the number of data decreases due to dimensions shrinking.

#### 4-2. LDA

1) LDA (Linear Discriminant Analysis) : Dimension reduction techniques similar to PCA

- Dimension reduction techniques applied in Supervised Learning
- Finding the axis that maximizes class(Label) separation

<br>
<center><img src="https://nirpyresearch.com/wp-content/uploads/2018/11/PCAvsLDA-1024x467.png" width="80%" height="80%"></center>
<center>[Comparision PCA vs LDA]</center>

##### [4-2-1] Example Code
<br>

PCA + LDA's Python code file is [hear](./ml%20algorithm/PCA_LDA.ipynb)

<br>

#### 4-3. Scaler
<br>

- One of Data-Preprocessing

- Each feature has its own range of data values. so if the range difference is large, it can converge to zero or diverge indefinitely when learning the model with the data.
   - Therefore, scaling allows you to adjust the data distribution or range of all features equally.

##### [4-3-1] Example Code

<br>

Python code file is [hear](./ml%20algorithm/Scaler.ipynb)

<br>

### 5. KNN (K-Nearest Neighbors)

<br>

1) Concept

Finding the 'nearest neighbor' is the model's prediction method (Classification).

- A type of Supervised Learning
- Need to apply Scaler Since it is judged on the basis of distance of each data.
- How to set k-values?

<br>

2) Theory 

   1. Calculate the distance between the new instance(Data to be predicted) from the known data
   2. Count the classes of the K Nearest Neighbors
   3. Classify the instance based on the majority of classes obtained in the previous step 
   4. Repeat steps 2. and 3. for all data(new instance)

<br>
<center><img src="https://s3.amazonaws.com/codecademy-content/courses/learn-knn/nearest_neighbor.gif" width="80%" height="80%"></center>
<center>[Process of KNN]</center>
<br>

#### [5-1] Cross Validation

- The most common strategy for judging the performance of a ML model is to divide the data-set into a training set and a validation set (at a rate of 70-80%) => **Holdout method**<br>
    - example : Get performance Split 5 only (below figure1)
    - problem : Datasets(not split5) might be possible to have a poor performance. So, we couldn't decide ML model by Holdout method score.

<br>

##### [5-1-1] K-fold Cross Validation
<br>
 - We can handle the problem mentioned above with K-fold cross validation.The dataset is divided into K pieces and the ML performance evaluation of k times is conducted. (The figure below shows K-fold cross validation with k value of 5)<br>
 - By evaluating all Train-Set, we can generalize the performance of the model.

<br>
<center><img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" width="80%" height="80%"></center>
<center>[figure1] simple diagram of K-fold Cross Validation</center>
<br>

- Conclusion : We have to divide the data-set into three types: <u>**train, validation, and test**</u>.

<br>

#### [5-2] Grid Search
<br>
 Grid-search provides the best parameters by sequentially entering hyperparameters used in classification or regression algorithms, learning and measuring. 

<br>

##### [5-2-1] GridSearchCV

 - A <u>total</u> search for the parameter values specified for the estimator

##### [5-2-2] RandomizedSearchCV

 - <u>Randomized</u> search for hyper parameters
 - **Unlike GridSearchCV**, not all parameter values are attempted, but a fixed number of parameter settings are sampled from the specified distribution. The number of parameter settings attempted is provided by `n_iter`.

<br>
<center><img src="https://www.researchgate.net/profile/Karl-Ezra-Pilario/publication/341691661/figure/fig2/AS:896464364507139@1590745168758/Comparison-between-a-grid-search-and-b-random-search-for-hyper-parameter-tuning-The.png" width="80%" height="80%"></center>
<center>[Diagram of Grid Search]</center>
<br>

#### [5-3] Example code

Python code file is [hear](./ml%20algorithm/KNN_Cross_Validation_Grid_Search.ipynb)

- how to build KMeans model
- how to set the K value? [Elbow method / Silhouette score]

<br>

### 6. NBC

### 7. DecisionTreeClassifier

<br>

- A type of Supervised Learning
- Express diagram of DecisionTree by `dtreeviz/graphviz` and so on.<br>
   - [[URL] Decision-Tree Visualization](https://mljar.com/blog/visualize-decision-tree/)
<br>
<center>

   ![image](../images/tree_visualization_r1.png)
</center>

- Easy to be **overfitting** ↔ RandomForest(DT + Bagging)

<br>

> class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

- criterion : 분할 품질을 측정하는 기능 (default : gini)
- max_depth : 트리의 최대 깊이 (값이 클수록 모델의 복잡도가 올라간다.)
- max_features : 각 노드에서 분할에 사용할 특징의 최대 수
- max_leaf_nodes : 리프 노드의 최대수

   [[Reference] DecisionTreeClassifier Hyperparameter](https://inuplace.tistory.com/548)

<br>

#### [7-1] Ensemble Learning

<br>

Ensemble Learning is the process where multiple machine learning models are combined to get better results. The core idea is that the result obtained from a combination of models can be more accurate than any individual machine learning model.

- Why we use Ensemble model? **Because it's powerful!** <br>
  Enemble shows better performance by merging each ML model that has lower performance.

<br>

#### [7-2] Bagging Classifier

<br>

- <u>A single ML model</u> learns to make individual predictions on a data-set sampled by **bootstrapping** and selects the final prediction result through voting
- Bagging : Bootstrap & Aggregation
   - Bootstrap : To extract N samples with **restore extraction** N times ↔ K-fold
   - Aggregation : Vote individual predictions (result : **value average**) to select final prediction results
- Representative model
   - **RandomForest**
   
   <br>
 - Steps<br>

   1. Create multiple data instances by dividing train data (bootstrapping)

   2. Create multiple models from this bootstrap data and multiple model outputs. Aggregate the results of the model and obtain the final results.

<br>
<center><img src="https://dezyre.gumlet.io/images/A+Comprehensive+Guide+to+Ensemble+Learning+Methods/Ensemble+learning+Bagging.png?w=900&dpr=1.3" width="70%" height="80%"></center>
<center>Diagram of Bagging Classifier</center>
<br>

##### [7-2-1] Random Forest
<br>

- Random Forest : Ensemble(Bagging type) of DecisionTree<br>
- Sampling both train data and feature → Lower **Overfitting**<br>
- Take large number of decision trees (`n_estimators : int`, default=100)

<br>
<center><img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png" width="70%" height="80%"></center>
<center>Diagram of Random Forest</center>
<br>

<br>

#### [7-3] Voting Classifier
<br>
- <u>Multiple different ML model</u> learn about the same data-set and vote to select the final prediction result with prediction results<br>
- Voting Classifiers refers to the "multiple classification", 
which can be divided in two methods : <u>Hard Voting Classifier and Soft Voting Classifier</u>

<br>

   - Hard Voting Classifier : Create multiple ML models and compare their performance about results. At this time, the result of the classifier is aggregated and the class that gets the most votes is determined as the final predicted value is called the Hard Voting Classifier. → **Majority  Vote**

<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile7.uf.tistory.com%2Fimage%2F997418435B42CEB012164F" width="70%" height="80%"></center>
<center>[figure2] Diagram of Hard Voting Classifier</center>
<br>

   ※ Conclusion : As shown above [figure2], the **final result**(prediction) of the Hard Voting Classifier will be <u>**1**</u> because there are three models that predict the final result as 1 and only one model that predict the final result as 2.
 
<br>

   - Soft Voting Classifier : Use when all classifiers used in an ensemble can predict the probability of a class. The prediction for each classifier is averaged to predict the class with the highest probability. → **Weighted Vote**

<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile23.uf.tistory.com%2Fimage%2F9922564D5B42D06106D981" width="50%" height="80%"></center>
<center>[figure3] Diagram of Soft Voting Classifier</center>
<br>

$p(i=1 | x) = \frac{(0.9 + 0.8 + 0.3 + 0.4)}4 = 0.6$
$p(i=2 | x) = \frac{(0.1 + 0.2 + 0.7 + 0.6)}4 = 0.4$

<br>

  ※ Conclusion : As shown above [figure2], the class with a high mean for the prediction probability (**Label 1**) is set as the final prediction class.

<br>

#### [7-4] Boosting

- To learn by **more weighting** <u>the next model</u> for incorrectly classified result values in the previous model (weak learner)
- Although **Bagging** is independent of individual models (but not completely with the same base estimator) ↔ **Boosting** is a <u>sequential Multiple classifier model</u>
- Representative model
   - Adaboost
   - Gradient boosting
      - XGBoost (https://velog.io/@dbj2000/ML)
      - LightGBM (more quick speed than XGBoost)
      - GBM (Gradient Boost Machine)
<br>

##### [7-4-1] GBM
<br>

- GBM : Gradient Boosting Machine
- GBM uses a **gradient descent method** for weight updates
- Hyperparameter Tunning
   - n_estimators, max_depth, Max_features : Number of trees
   - loss : Select Cost Function {‘log_loss’, ‘deviance’, ‘exponential’}
   - learning_rate : The coefficient applied by learner to correct the error values sequentially
   - subsample : Proportion of data used for learning

<br>

##### [7-4-2] XGBoost
<br>

- XGBoost : eXtra Gradient Boost → Tree-based ensemble learning.
- Advantage

<center>

|List|Content|
|:---:|:---:|
|Good Performance|Typically excellent predicton at classification and regression|
|Quick running time than GBM|Parallel execution and many functions ↔ Gradient Descent|
|Regulation in Overfitting|A function|

</center>

<br>

#### [7-5] Example code

Python code file is [hear](./ml%20algorithm/DecisionTree.ipynb)
