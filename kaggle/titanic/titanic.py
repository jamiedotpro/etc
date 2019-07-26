import os
import pandas as pd

dir_path = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(dir_path, 'input/train.csv')
test_file = os.path.join(dir_path, 'input/test.csv')
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

print('start-------------------------------------------')
#print('\n--train--\n', train.head())
#print('\n--test--\n', test.head())
print()
# 결측치 분석
#print('\n--train info--\n', train.info())
#print('\n--train.isnull.sum--\n', train.isnull().sum())

# Age : 결측치가 많지 않고 나이에 따라 생존 여부와 상관 있을 것으로 예상되어 데이터 채워 넣어야 함
# Cabin : 객실 번호가 생존 여부와 관련 있을 수 있으나 결측치가 너무 많기때문에 제거하는 것이 나을 것으로 보임
# Embarked : 2개의 결측치가 있으므로 어느 값으로 채워도 문제 없어 보임

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    #df.plot(kind='bar', stacked=True, figsize=(10,5), title=feature)
# bar_chart('Sex')
# bar_chart('Pclass')
# bar_chart('SibSp')
# bar_chart('Parch')
# bar_chart('Embarked')
# plt.show()

# 데이터 가공

# 나이 비었는지 확인
# age_nan_rows = train[train['Age'].isnull()]
# print(age_nan_rows.head())

from sklearn.preprocessing import LabelEncoder
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
test['Sex'] = LabelEncoder().fit_transform(test['Sex'])

# print(test.head())
train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train['Name'].unique()
# print(titles)
test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test['Name'].unique()
# print(test_titles)

train['Age'].fillna(-1, inplace=True)
test['Age'].fillna(-1, inplace=True)

medians = dict()
for title in titles:
    median = train.Age[(train['Age']!=-1) & (train['Name']==title)].median()
    medians[title] = median

for index, row in train.iterrows():
    if row['Age'] == -1:
        train.loc[index, 'Age'] = medians[row['Name']]

for index, row in test.iterrows():
    if row['Age'] == -1:
        test.loc[index, 'Age'] = medians[row['Name']]

print(train.head())
print(medians)
print(train.isnull().sum())
print(test.isnull().sum())

test_age_nan_rows = test[test['Age'].isnull()]
print(test_age_nan_rows)

# 이름별로 산사람과 죽은 사람을 비교해보자
fig = plt.figure(figsize=(15,6))
i=1
for title in train['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')
    i += 1
# plt.show()

title_replace = {
    'Don':0,
    'Rev':0,
    'Capt':0,
    'Jonkheer':0,
    'Mr':1,
    'Dr':2,
    'Major':3,
    'Col':3,
    'Master':4,
    'Miss':5,
    'Mrs':6,
    'Mme':7,
    'Ms':7,
    'Lady':7,
    'Sir':7,
    'Mlle':7,
    'the Countess':7
}
train['Name'].unique()
test['Name'].unique()
print(test[test['Name'] == 'Dona'])

train['Name'] = train['Name'].apply(lambda x: title_replace.get(x))
# print(train.head())
test['Name'] = test['Name'].apply(lambda x: title_replace.get(x))

# print(test.isnull().sum())
# print(test[test['Name'].isnull()])

test[test['Sex'] == 0]['Name'].mean()
train[train['Sex'] == 0]['Name'].mean()
test[test['Name'].isnull()]['Sex']
test[test['Name'].isnull()]['Name']
test['Name'] = test['Name'].fillna(value=train[train['Sex'] == 0]['Name'].mean())
print(test.head())
print('train.isnull().sum()', train.isnull().sum())
print('test.isnull().sum()', test.isnull().sum())

train_test_data = [train, test]
for dataset in train_test_data:
    dataset.loc[ dataset['Age']<=10, 'Age'] = 0,
    dataset.loc[(dataset['Age']>10)&(dataset['Age']<=16), 'Age'] = 1,
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=20), 'Age'] = 2,
    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=26), 'Age'] = 3,
    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=30), 'Age'] = 4,
    dataset.loc[(dataset['Age']>30)&(dataset['Age']<=36), 'Age'] = 5,
    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=40), 'Age'] = 6,
    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=46), 'Age'] = 7,
    dataset.loc[(dataset['Age']>46)&(dataset['Age']<=50), 'Age'] = 8,
    dataset.loc[(dataset['Age']>50)&(dataset['Age']<=60), 'Age'] = 9,
    dataset.loc[ dataset['Age']>60, 'Age'] = 10

fig = plt.figure(figsize=(15,6))
i = 1
for age in train['Age'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Age : {}'.format(age))
    train.Survived[train['Age'] == age].value_counts().plot(kind='pie')
    i += 1
# plt.show()

age_point_replace = {
    0: 8,
    1: 6,
    2: 2,
    3: 4,
    4: 1,
    5: 7,
    6: 3,
    7: 2,
    8: 5,
    9: 4,
    10: 0   
}

for dataset in train_test_data:
    dataset['age_point'] = dataset['Age'].apply(lambda x: age_point_replace.get(x))

# print('train.head()\n', train.head())
# print('test.head()\n', test.head())

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# print('train.head()\n', train.head())
# print('test.head()\n', test.head())


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

maybe_dad_mask = (train['FamilySize'] > 4) & (train['Sex'] == 1)
print(maybe_dad_mask.head())
train['maybe_dad'] = 1
train.loc[maybe_dad_mask,'maybe_dad'] = 0
train[train['maybe_dad'] == 0].head()

fig = plt.figure()
ax1 = train.Survived[train['maybe_dad'] == 1].value_counts().plot(kind='pie')
#plt.show()
ax2 = train.Survived[train['maybe_dad'] == 0].value_counts().plot(kind='pie')
#plt.show()
test['maybe_dad'] = 1
test_maybe_dad_mask = (test['FamilySize'] > 4) & (test['Sex'] == 1)
test.loc[test_maybe_dad_mask,'maybe_dad'] = 0
print(train['FamilySize'].unique())
print(test['FamilySize'].unique())

fig = plt.figure(figsize=(15,6))

i = 1
for size in train['FamilySize'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Size : {}'.format(size))
    train.Survived[train['FamilySize'] == size].value_counts().plot(kind='pie')
    i += 1

#plt.show()

size_replace = {
    1: 3,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 1,
    7: 4,
    8: 0,
    11: 0
}

for dataset in train_test_data:
    dataset['fs_point'] = dataset['FamilySize'].apply(lambda x: size_replace.get(x))
    dataset.drop('FamilySize',axis=1,inplace=True)

print(train.head())

# print(train.isnull().sum())
# print(test.isnull().sum())

fig = plt.figure(figsize=(15,6))

i = 1
for x in train['Pclass'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Pclass : {}'.format(x))
    train.Survived[train['Pclass'] == x].value_counts().plot(kind='pie')
    i += 1


for dataset in train_test_data:
    dataset.loc[dataset['Pclass']==3,'Pclass_point'] = 0
    dataset.loc[dataset['Pclass']==2,'Pclass_point'] = 1
    dataset.loc[dataset['Pclass']==1,'Pclass_point'] = 2

fig = plt.figure(figsize=(15,6))

i = 1
for x in train['Embarked'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Em : {}'.format(x))
    train.Survived[train['Embarked'] == x].value_counts().plot(kind='pie')
    i += 1

for dataset in train_test_data:
    dataset.loc[dataset['Embarked']==0,'Em_point'] = 0
    dataset.loc[dataset['Embarked']==2,'Em_point'] = 1
    dataset.loc[dataset['Embarked']==1,'Em_point'] = 2

print(train.isnull().sum())
print(test.isnull().sum())

print(train['Cabin'].unique())

for data in train_test_data:
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    data['Cabin'].unique()
    data['Fare'].fillna(0,inplace=True)
    data['Fare'] = data['Fare'].apply(lambda x: int(x))

fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1

    
temp = train['Fare'].unique()
temp.sort()
print('temp:\n', temp)

for dataset in train_test_data:
    dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0,
    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=80), 'Fare'] = 1,
    dataset.loc[(dataset['Fare']>80)&(dataset['Fare']<=100), 'Fare'] = 2,
    dataset.loc[(dataset['Fare']>100), 'Fare'] = 3

fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 0), 'Cabin'] = 'G',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 1), 'Cabin'] = 'T',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 2), 'Cabin'] = 'C',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 3), 'Cabin'] = 'B',

fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1

fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'G'), 'Cabin_point'] = 0,
    dataset.loc[(dataset['Cabin'] == 'C'), 'Cabin_point'] = 3,
    dataset.loc[(dataset['Cabin'] == 'E'), 'Cabin_point'] = 5,
    dataset.loc[(dataset['Cabin'] == 'T'), 'Cabin_point'] = 1,
    dataset.loc[(dataset['Cabin'] == 'D'), 'Cabin_point'] = 7,
    dataset.loc[(dataset['Cabin'] == 'A'), 'Cabin_point'] = 2,
    dataset.loc[(dataset['Cabin'] == 'B'), 'Cabin_point'] = 6,
    dataset.loc[(dataset['Cabin'] == 'F'), 'Cabin_point'] = 4,

fig = plt.figure(figsize=(15,6))

i=1
for x in train['Fare'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Fare : {}'.format(x))
    train.Survived[train['Fare'] == x].value_counts().plot(kind='pie')
    i += 1


for dataset in train_test_data:
    dataset.loc[(dataset['Fare'] == 0), 'Fare_point'] = 0,
    dataset.loc[(dataset['Fare'] == 1), 'Fare_point'] = 1,
    dataset.loc[(dataset['Fare'] == 2), 'Fare_point'] = 3,
    dataset.loc[(dataset['Fare'] == 3), 'Fare_point'] = 2,


from sklearn.preprocessing import StandardScaler
for dataset in train_test_data:
    dataset['Name'] = StandardScaler().fit_transform(dataset['Name'].values.reshape(-1, 1))
    dataset['Sex'] = StandardScaler().fit_transform(dataset['Sex'].values.reshape(-1, 1))
    dataset['maybe_dad'] = StandardScaler().fit_transform(dataset['maybe_dad'].values.reshape(-1, 1))
    dataset['fs_point'] = StandardScaler().fit_transform(dataset['fs_point'].values.reshape(-1, 1))
    dataset['Em_point'] = StandardScaler().fit_transform(dataset['Em_point'].values.reshape(-1, 1))
    dataset['Cabin_point'] = StandardScaler().fit_transform(dataset['Cabin_point'].values.reshape(-1, 1))
    dataset['Pclass_point'] = StandardScaler().fit_transform(dataset['Pclass_point'].values.reshape(-1, 1))
    dataset['age_point'] = StandardScaler().fit_transform(dataset['age_point'].values.reshape(-1, 1))
    dataset['Fare_point'] = StandardScaler().fit_transform(dataset['Fare_point'].values.reshape(-1, 1))

train.drop(['PassengerId','Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)
test.drop(['Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

print('train_data.head()\n', train_data.head())
print('target.head()\n', target.head())

print('test.shape: ', test.shape)
print('train.shape: ', train.shape)
print('train_data.shape: ', train_data.shape)

# svg 결과 만들기
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = SVC()
# clf = DecisionTreeClassifier()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission_test1.csv', index=False)
submission = pd.read_csv('submission_test1.csv')
print(submission.head())