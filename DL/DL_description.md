# Study of Machine-Learning

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.10/13

### Reference

- [[Github] Chahohyun [Neural Net Study]](https://github.com/ChaHoHyun/Neural_Net_Study)
- [[Github]Sungwookle [CNN]](https://github.com/SungwookLE/ReND_Car_TensorLab_with_NeuralNet/blob/master/2.Convolutional_Neural_with_LeNet_Study/Study_ConvNet.md)
- [How to Calculate Hidden Layer's Backpropagation](https://bskyvision.com/718)
- [About Activation Function](https://deepinsight.tistory.com/113)
- [About Gradient Descent](https://angeloyeo.github.io/2020/08/16/gradient_descent.html)
- [Funtional API vs Sequential API Wikidocs](https://wikidocs.net/38861)
- [[URL] What is CNN](https://yjjo.tistory.com/8)
- [[Youtube] What is Convolution](https://www.youtube.com/watch?v=9Hk-RAIzOaw&list=RDCMUCUR_LsXk7IYyueSnXcNextQ&index=1)
- [CNN Project - Image Classification](https://inhovation97.tistory.com/33)
   - [Web Crawling by Selenium](https://velog.io/@jungeun-dev/Python-%EC%9B%B9-%ED%81%AC%EB%A1%A4%EB%A7%81Selenium-%EA%B5%AC%EA%B8%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%88%98%EC%A7%91)
   - [★ keras CNN Example](https://codetorial.net/tensorflow/mnist_classification.html)
- [[URL] What is RNN?](https://codetorial.net/tensorflow/mnist_classification.html)

### Index

1. [Single / Hidden-Layer Perceptron](#1-perceptron)<br>
   [1-4] [Functional API](#1-4-funtional-api)
2. [CNN](#2-convolution-neural-network)
3. [RNN](#3-reccurent-neural-network)

<br>


<br>
<center><img src="https://pubs.acs.org/cms/10.1021/acs.analchem.0c04671/asset/images/medium/ac0c04671_0001.gif" width="70%" height="100%"></center>
<center>[The evolution of deep learning: from perceptron to CNN]</center>

### 1. Perceptron
<br>

#### [1-1] Single Neuron (ANN)

A single-layer neural network represents the most simple form of neural network, in which there is only one layer of input nodes that send weighted inputs to a subsequent layer of receiving nodes, or in some cases, one receiving node.

- ANN (Artificial Neural Network) : Machine learning algorithms imitating human neural network principles and structures (**Only Feed Forward**)
   - Limitation<br>
      1. It is difficult to find the optimal value of the parameter in the learning process.
      2. Overfitting
      3. The learning time is too slow.
<br>

#### Proof Reference
<br>

- MD file and Python code file is [[Github] URL hear](https://github.com/ChaHoHyun/Neural_Net_Study/blob/main/Summary_study_Neural_Net.md)

<br>
<center><img src="https://www.lgcns.com/wp-content/uploads/2021/11/99C360355E86DBB514.png" width="50%" height="100%"></center>
<center>[Human Brain vs Deep Learning]</center>
<br>
<center><img src="https://blog.kakaocdn.net/dn/ADQNA/btqNpHmaVSK/lRgSJKYPKRgkOtAQnBafuK/img.gif" width="50%" height="100%"></center>
<center><img src="https://gowrishankar.info/blog/do-you-know-we-can-approximate-any-continuous-function-with-a-single-hidden-layer-neural-network-a-visual-guide/single_neuron.png" width="50%" height="100%"></center>
<center>[Single Neuron]</center>
<br>

- [Example Code](./dl_algorithm/perceptron.ipynb)

- Limitation
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
   - Softmax : A activation function with the characteristic that the input value is normalized to all values between 0 and 1 as the sum of the output values is always 1
   > $Softmax(x_1) = \frac {e^{z_{1}}} {\sum \limits_{i=1}^{k} e^{z_{k}}} for\ i = 1,...,k$ 

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

         - Batch Size : The number of data to learn at once when updating<br>
         - Epoch : The number of times the entire training dataset has passed through the neural network<br>
         - Iteration : Number of times updated within 1 epoch
         
         <br>
         <center><img src="https://mblogthumb-phinf.pstatic.net/MjAxOTAxMjNfMjU4/MDAxNTQ4MjM1Nzg3NTA2.UtvnGsckZhLHOPPOBWH841IWsZFzNcgwZvYKi2nxImEg.CdtqIxOjWeBo4eNBD2pXu5uwYGa3ZVUr8WZvtldArtYg.PNG.qbxlvnf11/20190123_182720.png?type=w800" width="50%" height="100%"></center>
         <center>[Batch Size vs Epoch vs Iteration]</center>

      2. Optimizer
         - GD(gradient descent) : It's a method of calculating using **all the data**
         - SGD(Stochastic gradient descent) : 확률적 경사 하강법. It is a method of randomly extracting a data and updating the weight itself.
         
         <br>
         <center><img src="https://t1.daumcdn.net/cfile/tistory/999EA83359D86B6B0B" width="50%" height="100%"><img src="https://t1.daumcdn.net/cfile/tistory/9961913359D86B9833" width="47.6%" height="100%"></center>

         - Momentum :  It is a method of using momentum.
         - **Adam(Adaptive Moment Estimation)** : Popular optimizer

         <center><img src="http://i.imgur.com/2dKCQHh.gif?1" width="50%" height="100%"><img src="http://i.imgur.com/pD0hWu5.gif?1" width="47.6%" height="100%"></center>
<br>

#### [1-3] Deep Neural Net
<br>

1. Concept
- Single Perceptron : Classification / Regression like Machine Learing
- Multi-Layer Perceptron : `Single Perceptrion` + Non-linearity(XOR) + Back-Propagation(Increasing Accuracy)
- Deep Neural Net : `Multi-Layer Perceptron` + More deeper network(Increasing Accuracy) + Relu & Drop-Out(Prevent Overfitting)<br>

2. [Example code + Funtional API](./dl_algorithm/Multi_Layer_Perceptron.ipynb)

<br>

#### [1-4] Funtional API
<br>

1. Sequetial API Example
```python
model = models.Sequential([
    layers.Dense(units = 10, activation='sigmoid', input_shape = x_train[0].shape),
    layers.Dense(units = 6, activation='sigmoid'),
    layers.Dense(units = 1, activation='linear')
])
```
2. Funtional API Example
```python
inputs = Input(x_train[0].shape)
x1 = layers.Dense(units = 10, activation = 'sigmoid')(inputs)
x2 = layers.Dense(units = 6, activation = 'sigmoid')(x1)
outputs = layers.Dense(1, activation = 'linear')(x2)
```
3. Why we use `Funtional API`?
- Sequential APIs have limitations in creating **complex models**, such as sharing multiple layers or using different types of inputs and outputs. Now let's see Functional APIs (API) code, a way to create more complex models.

```python
inputs = Input(x_train[0, :4].shape)
x1 = layers.Dense(8, activation='sigmoid')(inputs)
x1 = layers.Dense(8, activation='sigmoid')(x1)
x1 = layers.Dense(4, activation='sigmoid')(x1)

x2 = layers.Dense(8, activation='sigmoid')(inputs)
x2 = layers.Dense(8, activation='sigmoid')(x2)
x2 = layers.Dense(4, activation='sigmoid')(x2)

x = layers.concatenate([x1,x2])
x = layers.Dense(4, activation='sigmoid')(x)
outputs = layers.Dense(3, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs)
```
<center>
<img src="./dl_algorithm/model_img/Funtional_model.png" width="45%" height="50%"> <img src="./dl_algorithm/model_img/Sequential_model.png" width="51.8%" height="50%"><br>
[Funtional API vs Sequential API]
</center>
<br>

#### [1-5] [Example code](./dl_algorithm/Multi_Layer_Perceptron.ipynb)
<br>
- Multi-Layer Perceptron + Functional API

<br>

### 2. Convolution Neural Network
<br>

#### [2-1] Concept

- It's one of the deep learning algorithm that imitates the human optic nerve.
- In particular, it maintains patial information of images by using convolution filter, dramatically reduces the amount of computation compared to fully connected neural networks(**DNN**), and shows good performance in image classification
<br>

#### [2-2] Basic knowledge

1. Image Data 
   - Black and white : Horizontal x Vertical x Black value
   - Color : Horizontal x Vertical x RGB value

<br>

<center>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fsm9GY%2Fbtqwoz5GP3S%2FrzMGckssOEeqbSvs7f0sd0%2Fimg.png" width="45%" height="50%"><br>
[About Image Data]
</center>
<br>

2. Convolution

   - definition : A mathematical funtion that multiplies one function by the value of the inversion of another function, and then integrates it over the interval to obtain a new function
   $(f∗g)(t)= \int_{∞}^{∞}f(τ)g(t−τ)dτ$

   - Field : Statistics, Computer vision, Natural language processing, Image processing, Signal processing, etc

   - Mathematical meaning : It means **extracting or filtering information** by locally amplifying or decreasing the signal using a kernel

   - Example ([Youtube Description](https://www.youtube.com/watch?v=9Hk-RAIzOaw&list=RDCMUCUR_LsXk7IYyueSnXcNextQ&index=1))

   <center><img src="https://velog.velcdn.com/images%2Fminchoul2%2Fpost%2F2011a19b-679e-4792-b9df-2444b6dd4606%2FConvolution_of_spiky_function_with_box2.gif" width="50%" height="50%"><center><img src="https://velog.velcdn.com/images%2Fminchoul2%2Fpost%2F203c3dc5-8a4e-428c-964b-6f1a25be1b4e%2FConvolution_of_box_signal_with_itself2.gif" width="50%" height="50%"><br>[Convolution Processing]</center>
<br>

3. Convolution Layer
- Use *Sliding window* method
- If the 4x4 image is flattened, *16 weights* must be learned for each characteristic. However, with the convolutional layers, the weights to learn have been reduced to *4 weights*, and the number of computational operations has been drastically reduced.
- Python code about conv filter is [hear](./dl_algorithm/Conv_filter.ipynb)
<br>
<center>
<img src="https://wikidocs.net/images/page/64066/conv15.png" width="45%" height="50%"><br>
[Convolution Layer]
</center>
<br>
<center><img src="https://camo.githubusercontent.com/1b19915bb05be7438cf5a511bad3ffe2df5942564eafb080bf0a710d228c259a/68747470733a2f2f6d626c6f677468756d622d7068696e662e707374617469632e6e65742f32303136303930395f3236332f636a683232365f31343733343136363032333033337a516b305f504e472f666967322e706e673f747970653d7732" width="50%" height="50%"><br>
[Types of Convolution Layer(filter)]</center>

<br>

#### [2-3] Structure of CNN
<br>

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnuUCk%2FbtqwsGRw5s4%2FxrKNIAY5HXx8d0tHK3z9Mk%2Fimg.png" width="50%" height="50%"><br>
[DNN(Fully Connected Layer) vs CNN]</center>
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FyhydK%2FbtqwmtFGpnh%2FgEfJAxzaEmud4beRKvh0gK%2Fimg.png" width="50%" height="50%"><br>
[Convolution Layer with bias]</center>
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbAFutP%2FbtqRAkU1Rq2%2FxZixvxgwJFONnszoh7up61%2Fimg.png" width="80%" height="50%"><br>
[Structure of Convolution Neural Network]</center>
<br>

- Hyper Parameter<br>
   1. Number of Convolution filter : It is recommended to balance the system while keeping the computation time/volume at each layer relatively constant.
   2. Padding status : It may not reduce the size of the input image.
   <br>
   <center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeJSFzX%2Fbtq9mnlb1Rr%2Fq8kgNEQuYm9QyxWh1PvUBk%2Fimg.png" width="60%" height="50%"><br>
   [Figure of Padding]</center>
   <br>

   3. filter size : Most **CNN** use 3x3 sizes over-lapping.
   4. Stride : Parameters that control the interval of movement of the filter 
   <br>
   <center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIYoOx%2Fbtq9gA0P3q7%2FQFst03O1TW8exBXtzWbYv1%2Fimg.png" width="60%" height="50%"><br>
   [Figure of Stride]</center>
   <br>
   5. Pooling layer type
   
   <br>

- Pooling Layer : The role of moderately reducing size and emphasizing specific features (Featuring)
   - Max Pooling (**Main** → [thesis] Boureau et al. ICML’10 for theoretical analysis )
   <br>
   <center><img src="https://blog.kakaocdn.net/dn/CAchP/btqwrS4GWfC/skXqkaRNx3u64zD5vJMXi1/img.gif" width="40%" height="50%"><br>
   [Figure of Stride]</center>
   <br>

   - Average Pooling
   <br>
   <center><img src="https://blog.kakaocdn.net/dn/p8PUR/btqwrAQI35e/cRxDmX83i8bvhh0WYafuX1/img.gif" width="40%" height="50%"><br>
   [Figure of Stride]</center>
   <br>

   - Min Pooling

<br>

- Flattening Layer : After extracting the feature through *Convolutional Layer and Pooling layer*, it is a step to connect the extracted characteristics to the Output layer for learning Network. Therefore, after the `flattening layer`, it is the same as a regular neural network model.
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcvouDZ%2FbtqwmuCU4dY%2FlmDRgbtnqqaiSOXrNyZLX1%2Fimg.png" width="30%" height="30%"><br>
[Figure of Flattening]</center>
<br>


- Summary
<br>

   1. Convolution
      - Input Data : $W_1×H_1×D_1$ ($W_1$: input width, $H_1$: input height, $D_1$: input channel or Depth)
      - Hyper Parameters
         - Number of filter : $K$
         - filter size : $F$
         - stride : $S$
         - padding : $P$
      - Output Data
         - $W_2=(W_1−F+2P)/S+1$
         - $H_2=(H1−F+2P)/S+1$
         - $D_2=K$ ($D_2$: Number of activation map)
      - Number of weight : $[F^2×D_1+D_1]×K$<br>
<br>
   2. Pooling
      - Input Data : $W_1×H_1×D_1$ ($W_1$: input width, $H_1$: input height, $D_1$: input channel or Depth)
      - Hyper Parameters
         - Number of filter : $K$
         - filter size : $F$
         - stride : $S$
         - padding : $P$
      - Output Data
         - $W_2=(W_1−F)/S+1$
         - $H_2=(H1−F)/S+1$
         - $D_2=D_1$

<br>

### 3. Reccurent Neural Network

<br>

#### [3-1] Concept

- RNN is a `sequence model` that processes inputs and outputs in sequence units
   - Sequence Data : Voice, Natural Language, Stock price

<br>

#### [3-2] Basic knowledge

1. Time Series Data 
   - Black and white : Horizontal x Vertical x Black value
   - Color : Horizontal x Vertical x RGB value

<br>

<center>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fsm9GY%2Fbtqwoz5GP3S%2FrzMGckssOEeqbSvs7f0sd0%2Fimg.png" width="45%" height="50%"><br>
[About Image Data]
</center>
<br>

2. Convolution

   - definition : A mathematical funtion that multiplies one function by the value of the inversion of another function, and then integrates it over the interval to obtain a new function
   $(f∗g)(t)= \int_{∞}^{∞}f(τ)g(t−τ)dτ$

   - Field : Statistics, Computer vision, Natural language processing, Image processing, Signal processing, etc

   - Mathematical meaning : It means **extracting or filtering information** by locally amplifying or decreasing the signal using a kernel

   - Example ([Youtube Description](https://www.youtube.com/watch?v=9Hk-RAIzOaw&list=RDCMUCUR_LsXk7IYyueSnXcNextQ&index=1))

   <center><img src="https://velog.velcdn.com/images%2Fminchoul2%2Fpost%2F2011a19b-679e-4792-b9df-2444b6dd4606%2FConvolution_of_spiky_function_with_box2.gif" width="50%" height="50%"><center><img src="https://velog.velcdn.com/images%2Fminchoul2%2Fpost%2F203c3dc5-8a4e-428c-964b-6f1a25be1b4e%2FConvolution_of_box_signal_with_itself2.gif" width="50%" height="50%"><br>[Convolution Processing]</center>
<br>

3. Convolution Layer
- Use *Sliding window* method
- If the 4x4 image is flattened, *16 weights* must be learned for each characteristic. However, with the convolutional layers, the weights to learn have been reduced to *4 weights*, and the number of computational operations has been drastically reduced.
- Python code about conv filter is [hear](./dl_algorithm/Conv_filter.ipynb)
<br>
<center>
<img src="https://wikidocs.net/images/page/64066/conv15.png" width="45%" height="50%"><br>
[Convolution Layer]
</center>
<br>
<center><img src="https://camo.githubusercontent.com/1b19915bb05be7438cf5a511bad3ffe2df5942564eafb080bf0a710d228c259a/68747470733a2f2f6d626c6f677468756d622d7068696e662e707374617469632e6e65742f32303136303930395f3236332f636a683232365f31343733343136363032333033337a516b305f504e472f666967322e706e673f747970653d7732" width="50%" height="50%"><br>
[Types of Convolution Layer(filter)]</center>

<br>

#### [2-3] Structure of CNN
<br>

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnuUCk%2FbtqwsGRw5s4%2FxrKNIAY5HXx8d0tHK3z9Mk%2Fimg.png" width="50%" height="50%"><br>
[DNN(Fully Connected Layer) vs CNN]</center>
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FyhydK%2FbtqwmtFGpnh%2FgEfJAxzaEmud4beRKvh0gK%2Fimg.png" width="50%" height="50%"><br>
[Convolution Layer with bias]</center>
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbAFutP%2FbtqRAkU1Rq2%2FxZixvxgwJFONnszoh7up61%2Fimg.png" width="80%" height="50%"><br>
[Structure of Convolution Neural Network]</center>
<br>

- Hyper Parameter<br>
   1. Number of Convolution filter : It is recommended to balance the system while keeping the computation time/volume at each layer relatively constant.
   2. Padding status : It may not reduce the size of the input image.
   <br>
   <center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeJSFzX%2Fbtq9mnlb1Rr%2Fq8kgNEQuYm9QyxWh1PvUBk%2Fimg.png" width="60%" height="50%"><br>
   [Figure of Padding]</center>
   <br>

   3. filter size : Most **CNN** use 3x3 sizes over-lapping.
   4. Stride : Parameters that control the interval of movement of the filter 
   <br>
   <center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIYoOx%2Fbtq9gA0P3q7%2FQFst03O1TW8exBXtzWbYv1%2Fimg.png" width="60%" height="50%"><br>
   [Figure of Stride]</center>
   <br>
   5. Pooling layer type
   
   <br>

- Pooling Layer : The role of moderately reducing size and emphasizing specific features (Featuring)
   - Max Pooling (**Main** → [thesis] Boureau et al. ICML’10 for theoretical analysis )
   <br>
   <center><img src="https://blog.kakaocdn.net/dn/CAchP/btqwrS4GWfC/skXqkaRNx3u64zD5vJMXi1/img.gif" width="40%" height="50%"><br>
   [Figure of Stride]</center>
   <br>

   - Average Pooling
   <br>
   <center><img src="https://blog.kakaocdn.net/dn/p8PUR/btqwrAQI35e/cRxDmX83i8bvhh0WYafuX1/img.gif" width="40%" height="50%"><br>
   [Figure of Stride]</center>
   <br>

   - Min Pooling

<br>

- Flattening Layer : After extracting the feature through *Convolutional Layer and Pooling layer*, it is a step to connect the extracted characteristics to the Output layer for learning Network. Therefore, after the `flattening layer`, it is the same as a regular neural network model.
<br>
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcvouDZ%2FbtqwmuCU4dY%2FlmDRgbtnqqaiSOXrNyZLX1%2Fimg.png" width="30%" height="30%"><br>
[Figure of Flattening]</center>
<br>


- Summary
<br>

   1. Convolution
      - Input Data : $W_1×H_1×D_1$ ($W_1$: input width, $H_1$: input height, $D_1$: input channel or Depth)
      - Hyper Parameters
         - Number of filter : $K$
         - filter size : $F$
         - stride : $S$
         - padding : $P$
      - Output Data
         - $W_2=(W_1−F+2P)/S+1$
         - $H_2=(H1−F+2P)/S+1$
         - $D_2=K$ ($D_2$: Number of activation map)
      - Number of weight : $[F^2×D_1+D_1]×K$<br>
<br>
   2. Pooling
      - Input Data : $W_1×H_1×D_1$ ($W_1$: input width, $H_1$: input height, $D_1$: input channel or Depth)
      - Hyper Parameters
         - Number of filter : $K$
         - filter size : $F$
         - stride : $S$
         - padding : $P$
      - Output Data
         - $W_2=(W_1−F)/S+1$
         - $H_2=(H1−F)/S+1$
         - $D_2=D_1$

<br>

<br><br><br><br><br><br>

$y_{i} = \beta_{0} + \beta_{1}x_{i}+\epsilon_{i}$<br>
$\epsilon_{i} = y_{i}-\beta_{0} - \beta_{1}x_{i}$
 > LSE → $MinS^2 = Min \sum \limits_{i=1}^{n} \epsilon_{i}^2 = Min \sum \limits_{i=1}^{n} (y_{i}-\beta_{0} - \beta_{1}x_{i})^2$

 - condition<br>
 
    1) $f^\prime(x) = 0$ 
       -  $\beta_{1} = \frac{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum \limits_{i=1}^{n}(x_{i}-\bar{x})^2} = \frac{Cov(X,Y)}{Var(X)}$
       -  $\beta_{0} = \bar{y} - \beta_{1}\bar{x}$
       
       <br>
    2) $f''(x) > 0$ => $Satisfied$


https://inhovation97.tistory.com/39?category=920763

https://velog.io/@jaehyeong/CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%9E%90%EB%8F%99%EC%B0%A8-%EC%82%AC%EA%B3%A0-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98

https://buomsoo-kim.github.io/keras/2018/05/02/Easy-deep-learning-with-Keras-8.md/

https://kdeon.tistory.com/44

https://eremo2002.tistory.com/76

