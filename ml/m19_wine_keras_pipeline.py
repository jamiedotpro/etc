import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# 데이터 읽어 들이기
dir_path = os.getcwd()
winequality_white_file = os.path.join(dir_path, 'etc/data/winequality-white.csv')
wine = pd.read_csv(winequality_white_file, sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# 숫자가 일치하지 않아서 한번 거치고 원핫인코딩함
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)


# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.0, optimizer='adam', node1=30, layer_cnt=1):
    inputs = Input(shape=(11, ), name='input')
    x = Dense(node1, activation='relu', name='hidden1')(inputs)
    x = Dense(16, activation='relu', name='hidden2')(x)
    for i in range(layer_cnt):
        x = Dense(8, activation='relu')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [5,10,20,30]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.0, 0.4, 5)
    node1 = [8,20,30,40,50,60]
    layer_cnt = [1,2,3,4,5]

    hyper_param = {'kcf__batch_size':batches,
                    'kcf__optimizer':optimizers,
                    'kcf__keep_prob':dropout,
                    'kcf__node1':node1,
                    'kcf__layer_cnt':layer_cnt}

    return hyper_param

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
pipe = Pipeline([('standardscaler', StandardScaler()), ('kcf', model)])

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

model = build_network(model_dic['kcf__keep_prob'], model_dic['kcf__optimizer'], model_dic['kcf__node1'], model_dic['kcf__layer_cnt'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=model_dic['kcf__batch_size'], epochs=500, callbacks=[early_stopping], verbose=1)

print('loss, acc: ', model.evaluate(x_test, y_test))
# loss, acc:  [0.2682528607699336, 0.9275510199215947]

