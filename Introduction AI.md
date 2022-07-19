# Study of AI

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.7/5

------------------------------------------------------------------------
- Background and Theory [Github](https://github.com/SungwookLE/ReND_Car_TensorLab_with_NeuralNet)
- [Linear Regression](https://datalabbit.tistory.com/49)
- [Linear Classifier](https://techdifferences.com/difference-between-linear-and-logistic-regression.html)
- [Neural Network BackPropagation](https://bskyvision.com/718)

### How to Execute

1. Create CSV data file by `data_generator.py`
2. Excute Neural-Net python file

- Single Perceptron (Only Out Layer) : `SingleLayer_1node.py`
- 2-Layer (h1, h2 : Hidden, o1 : Output Layer) : `twolayer_2node.py`

3.  Visualization by plt.show()

## [1] What is AI?
<br>

### **1. Concept of AI**
<br>
<center><img src="https://www.codingdojo.com/blog/wp-content/uploads/ai-v2-img3.jpg" width="70%" height="70%"></center>
<br>

   AI stands for 'Artificial Intelligence'. The developement of AI is primarily dependent on 1) **access of available large amounts of data**, 2) **the evolution of technology allowing the data process**(such as Semiconductor) and 3) **manipulation better than humans**.

 There are two types of artificial intelligence.

 1. Weak AI(Narrow AI) : Weak AI that performs a given task in one particular field according to human instructions
    - For example :  Searching the internet, Disease detection, Facial recognition, Recommender systems etc.
 2. Strong AI(General AI) : Strong AI can think and act like a human. Normal AI is where we're headed, but it's still in its early stages.
    - For example :  Chatbots, Autonomous Vehicles etc.
<br>

### **2. AI vs Machine Learning vs Deep Learning**
<br>

 Deffence of AI, ML, DL : We usually divide **Artificial Intelligence** into AI(Rule-Based AI), Machine Learning(ML), and Deep Learning(DL).
<br>
<center><img src="https://nealanalytics.com/wp-content/uploads/2020/03/AI-ML-DL-Diagram.png" width="90%" height="100%"></center>
<br>

<center>

![image](./images/difference%20of%20Ai.png)
</center>

1) Rule-Based AI : A system designed to achieve artificial intelligence through a model based **only on predetermined rules** is called a rule-based AI system.
 - What is limit? 
    1) Need to handle all exceptions
    2) The logic implementer(coder) must be aware of the full extent of the domain
2) ML(Machine Learning) : Human teaches machines how to make inferences and decisions based on past experience. Identify patterns and analyze historical data to guess what these data mean, and reach possible conclusions without requiring human experience. This machine learning can help you evaluate data and draw conclusions, saving your business time and making better decisions.
  - What is limit? Humans should directly process and inform the data so that the machine can understand it more easily(**Feature Engineering**).
  - For Example : Linear/Logistic Regression, k-Nearest Neighbors, Decision trees & Random forests, Support Vector Machine(SVM), Neural networks and so on.
3) DL(Deep Learning) : It's a kind of ML method. To classify and guess the results, we teach the machine to process inputs through layers
  - For Example : CNN, RNN, LSTM, RBM... + AlphaGo by DeepMind

### **3. Type Machine learning**

There are **three methods** machine learning in large categories. It is unsupervised learning, supervised learning, and reinforcement learning
<center><img src="https://miro.medium.com/max/2000/1*8wU0hfUY3UK_D8Y7tbIyFQ.png" width="90%" height="100%"></center>

[Reference] [Machine Learning Types - Scikitlearn](https://scikit-learn.org/0.15/user_guide.html)

 - Supervised learning : Learning a model from **labeled training data** to make predictions about future data that you have never seen before. Therefore, Supervised learning creates a function model that get outputs data with input data
   1) classification : Classification refers to the problem of classifying a given data according to a given category (label). ex) Yes or No, Apple or Grape
   2) regression : The problem of predicting continuous values based on the feature of the data
  <center><img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/Regression_vs_Classification.jpg" width="70%" height="100%"></center><br>

 - Unsupervised learning : To predict results for new data by clustering data **without correct answer labels** among similar features. Therefore, it is possible to understand the **probability distribution** of the next data by the existing data.
   1) Clustering
   2) Feature Extraction (Visulization, Dimensionality reduction)
   3) Model Generation(ex dalle2)
 
 <br>
 - Reinforcement learning : Optimized Method by continuous experience (Trial and Error) and evaluating (Rewards & Penalty)

<br>

### 4. methodology
<br>

#### [1] Data Pre-processing
##### [1-1] Remove missing values(NaN)

#### [2] Validation & Evaluation

#### [3] Normalization & Standardization

#### [4] Overfitting vs Underfitting

#### [5] AI Modeling
##### [5-1] generate model
##### [5-2] model fit
##### [5-3] predict x_test
##### [5-4] validate results
<br>

<center><img src="https://miro.medium.com/max/1396/1*lARssDbZVTvk4S-Dk1g-eA.png" width="80%" height="100%"></center>
<center>[Overfitting & Underfitting diagram]</center>  
<br>
<center><img src="https://vitalflux.com/wp-content/uploads/2020/12/overfitting-and-underfitting-wrt-model-error-vs-complexity.png" width="70%" height="100%"></center>
<center>[Performance of Overfitting vs Underfitting]</center>  

