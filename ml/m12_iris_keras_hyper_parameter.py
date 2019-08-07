from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

import numpy
import pandas as pd
import os
import tensorflow as tf

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8', header=None,
                        names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'])

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
# loc는 레이블로 분리
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

'''
parameters = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.0001],
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)


y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)
'''

# 문자열 데이터여서 One-hot encoding 할 수 있게 숫자로 바꿔줌
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encoding...
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(4, ), name='input')
    x = Dense(8, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    prediction = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [1,5,10,20]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.0, 0.4, 5)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용

search.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
print('predict: \n', search.predict(x_test))

