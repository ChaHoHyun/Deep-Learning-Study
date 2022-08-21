# Study about AI

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.8/21

### Introduction

<br>

- Understand what is the our goal and what each features, data mean.

- Understand how to evaluate ML performance.

<br>

### Import Machine-Learning Library
<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
plt.rc("figure", figsize=(12,10))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.svm import SVC
```
<br>

### Collecting CSV file
<br>

```python
train = pd.read_csv("file path/train.csv")
test = pd.read_csv("file path/test.csv")

print(train.shape, test.shape)
print(train.info(), test.info())
print(train.isnull().sum(), test.isnull().sum())
```
<br>

### EDA & Visualization & Feature Engineering
<br>

```python
# visualization
train.loc[train[col_name] == 1].plot(kind='kde') # Same with `sns.distplot()`
train.loc[train[col_name] == 2].plot(kind='kde')
df.plot(kind='bar', stacked=True, figsize=(10,8))

# set plot
plt.xlim([0, 80]) # change x axis length of graph
plt.xticks(rotation=0) # change xticks angle

dataset = [train, test]

column5_mapping = {"feature1" : 0,
                    "feature2" : 1}

for data in dataset:
    data[column1].fillna(data.groupby(column2)[column1].transform('median'), inplace=True)
    
    data.dropna([column3, column4], inplace=True)

    data[column5].map(column5_mapping)

    data.drop(columns=['cols_name1', 'cols_name2'], axis=1, inplace=True)

dummies = pd.get_dummies(train['cols_name'], prefix='cols_name')
train = pd.concat([train, dummies], axis=1)
```
<br>

### ML Modeling
<br>

```python
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model_list = [DT(), KNN(n_neighbors=5, n_jobs=-1), RFC(), GBM(), SVC()]

for model in model_list:
    score = cross_val_score(model, x_train, y_train, 
                            cv=k_fold, n_jobs=-1, scoring='accuracy').mean()
    print(f"{str(clf)} Score : {np.round(score,2)}")

clf = RFC(n_jobs=-1, random_state=0)
clf.fit(x_train, y_train)

test = test.drop(columns='PassengerId')

result = clf.predict(x_test)
```
<br>

### Final
<br>

```python
submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    'Survived' : result
})
submission.to_csv('submission.csv', index=False)

submission = pd.read_csv("./submission.csv")
submission
```
