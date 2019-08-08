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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


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
    hyper_param = {'kcf__batch_size':batches,
                    'kcf__optimizer':optimizers,
                    'kcf__keep_prob':dropout}

    return hyper_param

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('kcf', model)])

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=pipe,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용

search.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
print('predict: \n', search.predict(x_test))

model_dic = search.best_params_

model = build_network(model_dic['kcf__keep_prob'], model_dic['kcf__optimizer'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=model_dic['kcf__batch_size'], epochs=500, callbacks=[early_stopping], verbose=0)

print('loss, acc: ', model.evaluate(x_test, y_test))
# loss, acc:  [0.07319126278162003, 0.9666666388511658]