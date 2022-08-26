# Study about AI

> **AUTHOR** : HoHyun Cha (ghgus2006@naver.com)  
> **DATE** : '22.8/21

<br>

### Index

1. [Import Library](#import-machine-learning-library)
2. [Collect CSV file](#collecting-csv-file)
3. [EDA & Viualization & Feature Engineering](#eda)<br>
    3-1. [[URL] Matplotlib Manual](https://wikidocs.net/159830)
4. [ML Modeling](#ml-modeling)

<br>

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
# plt.rc("figure", figsize=(12,10))

from sklearn.metrics import accuracy_score, mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict

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

### EDA
<br>

- Explore Data

```python
df['col_name'].isna().sum()
df['col_name'].value_counts()

pd.set_option('display.max_rows', 10000) # Pandas Setting
```
<br>
- Datetime

```python
pd.to_datetime(train['datetime']) # change data type : str to datetime

data['year'] = data['datetime'].dt.year # month, day, hour, minute, second
data["weekday"] = data["datetime"].dt.day_name()
data["weekday_int"] = data["datetime"].dt.dayofweek
```
<br>
- One Hot Encoding

```python
dummies = pd.get_dummies(train['cols_name'], prefix='cols_name')
train = pd.concat([train, dummies], axis=1)
# or

(train['holiday'] == 0) & (train['workingday'] == 0) # True/False
```
<br>

#### Visualization
<br>

- Correlation heat map

```python
cols = list(train.columns)
corr = train[cols].corr()
mask = np.array(corr)

plt.figure(figsize=(20,20))
sns.heatmap(corr, vmax=0.8, square=True, annot=True, cmap="coolwarm")
```
- Subplot

```python
plt.subplots(figsize=(20,12))

col_list = ['year', 'month', 'day', 'hour']

for i in range(len(col_list)):
    plt.subplot(2,2,i+1)
    
    col_uni = train[col_list[i]].unique()
    colors = sns.color_palette('hls',len(col_uni))
    
    train.groupby(col_list[i])['count'].mean().plot(kind='bar', color=colors)
    plt.xlabel(col_list[i], fontsize=20, weight='bold')
    plt.ylabel('Count', fontsize=20, weight='bold')
    plt.xticks(fontsize=14, rotation=0)
```
=
```python
figure, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,12))

colors = sns.color_palette('hls',len(train['year'].unique()))

train.groupby('year')['count'].mean().plot(kind='bar', ax=ax1, color=colors)
ax1.set_xlabel('year', fontsize=20, weight='bold')
ax1.set_ylabel('Count', fontsize=20, weight='bold')
ax1.set_xticklabels(train['year'].unique(),fontsize=14, rotation=0)
# Month, Day, Hour is Same
```
- Figure Setting
```python
train.loc[train[col_name] == 1].plot(kind='kde') # Same with `sns.distplot()`
train.loc[train[col_name] == 2].plot(kind='kde')
# Histogram : Same with 'kde
train['windspeed_enc'].plot.hist(bins=20,alpha=0.5,color=['green','blue'], figsize=(10,8))

df.plot(kind='bar', stacked=True, figsize=(10,8))

df.plot.scatter(x='temp', y='atemp', figsize=(10,8),s=10, colormap='hot')

# set plot
plt.xlim([0, 80]) # change x axis length of graph
plt.xticks(rotation=0) # change xticks angle
plt.xlabel('ylabel', fontsize=20, weight='bold')
```
<br>

- Seaborn ('hue' function)
```python
plt.figure(figsize=(10,8))

sns.barplot(data = train, x = "year", y = 'count')
sns.pointplot(data=train, x="hour", y="count")
sns.scatterplot(data=train, x="temp", y="atemp", hue="windspeed", size="count", sizes=(0, 150))
sns.countplot(data=train, x='weather')
sns.distplot(train["windspeed"])
```
<br>

#### Feature Engineering
<br>

- apply
```python
def make_year_month(df):
    return df.strftime("%Y%m")

train['datetime'].apply(make_year_month)
```
- transform / mapping / Drop 
```python
dataset = [train, test]

column5_mapping = {"feature1" : 0,
                    "feature2" : 1}

for data in dataset:
    data[column1].fillna(data.groupby(column2)[column1].transform('median'), inplace=True)
    
    data.dropna([column3, column4], inplace=True)

    data[column5].map(column5_mapping)

    data.drop(columns=['cols_name1', 'cols_name2'], axis=1, inplace=True)

train[train.columns.difference(['casual', 'registered','count'])]
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
    print(f"{str(model)} Score : {np.round(score,2)}")

clf = RFC(n_jobs=-1, random_state=0)
clf.fit(x_train, y_train)

test = test.drop(columns='PassengerId')

result = clf.predict(x_test)
```
<br>

- Regression Solution
```python
from sklearn.ensemble import RandomForestRegressor

y_train_count_log = np.log(y_train_count + 1)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

model = RandomForestRegressor(n_estimators = 1000, n_jobs=-1)

y_predict_count_log = cross_val_predict(model, x_train, y_train_count_log, cv=k_fold, n_jobs=-1)
y_predict_count = np.exp(y_predict_count_log) - 1
y_predict = np.sqrt(y_predict_count**2)
score = mean_squared_log_error(y_train_count, y_predict)
score = np.sqrt(score)
print("Score= {0:.5f}".format(score))
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
