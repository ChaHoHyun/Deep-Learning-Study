# Study of Machine-Learning

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.10/13

### Reference

- [Github - Chahohyun [Neural Net Study]](https://github.com/ChaHoHyun/Neural_Net_Study)
- [Github - Sungwookle [CNN]](http://sungwookle.site/research/2106211010/)
- [How to Calculate Hidden Layer's Backpropagation](https://bskyvision.com/718)
- [About Activation Function](https://deepinsight.tistory.com/113)
- [About Gradient Descent](https://angeloyeo.github.io/2020/08/16/gradient_descent.html)

### Index

1. [Single / Hidden-Layer Perceptron](#1-perceptron)
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

### 1. Perceptron
<br>

#### [1-1] Single Neuron

A single-layer neural network represents the most simple form of neural network, in which there is only one layer of input nodes that send weighted inputs to a subsequent layer of receiving nodes, or in some cases, one receiving node.

<br>
<center><img src="https://www.lgcns.com/wp-content/uploads/2021/11/99C360355E86DBB514.png" width="50%" height="100%"></center>

<center>[Human Brain vs Deep Learning]</center>
<br>

<center><img src="https://gowrishankar.info/blog/do-you-know-we-can-approximate-any-continuous-function-with-a-single-hidden-layer-neural-network-a-visual-guide/single_neuron.png" width="50%" height="100%"></center>
<center>[Single Hidden Layer Neural Network]</center>
<br>

##### Limitation
<br>

<center><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOgA-qClvpVbeTSSPZm2y64vSdvDBtfvzgLw&usqp=CAU" width="50%" height="100%"></center>
<center>[XOR Problem]</center>

#### [1-2] Hidden Layer
<br>

By adding hidden layer composed of several perceptrons to an artificial neural network, the accuracy and complexity is much improved *only training set*. = We can classifier Label by a lot of line by creating hidden layer.

<center><img src="https://user-images.githubusercontent.com/57344945/103791132-6dec3080-5085-11eb-9f7c-52886b3d1bb2.png" width="60%" height="100%"></center>
<center>[More Hidden layer diagram]</center>
<br>

- Activation Function : A function that converts the sum of input signals into output signals

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvnQPJ%2FbtqEd7HdyUh%2FETBAZp5B17K8KiwAleT3HK%2Fimg.png" width="50%" height="100%"></center>
<center>[Sort of Activation Functions]</center>

In fact, the most used is `ReLU`. <br>
Why? Because of **Back-Propagation (Gradient Vanishing)**. Through backpropagation we update the weights and biases, using the derivative of the cost function. However the derivative function of sigmoid has a value of 0 in a specific interval, causing a problem in that the update amount is lost. In other words, there is a problem in that learning cases at certain moment.

<br>

- Cost Function : the overall degree of error that exists in the training data.
   - MSE (Mean Squared Error) : $\frac{1}{n} \sum \limits_{i=1}^{n} ((y_{i} - \hat{y_{i}})^2)$
   - Binary Cross-entropy : $-\frac{1}{n} \sum \limits_{i=1}^{n} (y_{i} \cdot log(\hat{y_{i}}) +(1-y_{i}) \cdot log(1-\hat{y_{i}}))$

- Gradient Descent(Optimizer) : How to find the optimized parameter that minimizes the cost value of the `cost function`<br>

   $\theta_{Update} =  \theta_{Old} - η \cdot \frac {\partial J(\theta_{0}, \theta_{1})}{\partial \theta_{j}}$<br>
   $η = Learning Rate$<br>
   - Precautions<br>
      1) Appropriately sized Learning Rate<br>
      <img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2020-08-16-gradient_descent/pic4.png" width="60%" height="100%">
      2) local minima problem<br>
      <img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2020-08-16-gradient_descent/pic5.png" width="30%" height="100%">
   
   <br>

   - Sort of Gradient Descent<br>
      <center><img src="https://t1.daumcdn.net/cfile/tistory/993D383359D86C280D" width="50%" height="100%"></center>
      <center>[Sort of Gradient Descent]</center>

      1. Terms<br>

         - Batch Size : How many data do you have at a time when updating<br>

      2. Optimizer
         - GD(gradient descent) : It's a method of calculating using **all the data**
         - SGD(Stochastic gradient descent) : 확률적 경사 하강법. It is a method of randomly extracting a data and updating the weight itself.
         
         <br>
         <center><img src="https://t1.daumcdn.net/cfile/tistory/999EA83359D86B6B0B" width="50%" height="100%"><img src="https://t1.daumcdn.net/cfile/tistory/9961913359D86B9833" width="47.6%" height="100%"></center>

         - Momentum :  It is a method of using momentum.
         - **Adam(Adaptive Moment Estimation)** : Popular optimizer

         <center><img src="http://i.imgur.com/2dKCQHh.gif?1" width="50%" height="100%"><img src="http://i.imgur.com/pD0hWu5.gif?1" width="47.6%" height="100%"></center>

<br>

#### [1-3] Proof Reference

MD file and Python code file is [[Github] hear](https://github.com/ChaHoHyun/Neural_Net_Study/blob/main/Summary_study_Neural_Net.md)

<br>




$y_{i} = \beta_{0} + \beta_{1}x_{i}+\epsilon_{i}$<br>
$\epsilon_{i} = y_{i}-\beta_{0} - \beta_{1}x_{i}$
 > LSE → $MinS^2 = Min \sum \limits_{i=1}^{n} \epsilon_{i}^2 = Min \sum \limits_{i=1}^{n} (y_{i}-\beta_{0} - \beta_{1}x_{i})^2$

 - condition<br>
 
    1) $f^\prime(x) = 0$ 
       -  $\beta_{1} = \frac{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})^2} = \frac{Cov(X,Y)}{Var(X)}$
       -  $\beta_{0} = \bar{y} - \beta_{1}\bar{x}$
       
       <br>
    2) $f''(x) > 0$ => $Satisfied$