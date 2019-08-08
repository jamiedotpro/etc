from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dir_path = os.getcwd()
pima_indians_file = os.path.join(dir_path, 'etc/data/pima-indians-diabetes.csv')
dataset = np.loadtxt(pima_indians_file, delimiter=',')
# dataset = np.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


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

# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(8, ), name='input')
    x = Dense(64, activation='relu', name='hidden1')(inputs)
    x = Dense(64, activation='relu', name='hidden2')(x)
    x = Dense(64, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [5,10,20,30]
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
# make_pipeline과 pipeline의 차이는 디폴트로 지정된 키 이름을 사용할지, 키 이름을 직접 입력할지의 차이임
# pipe = make_pipeline(MinMaxScaler(), model) # == pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('kerasclassifier', model)])
pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('kcf', model)])

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=pipe,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용
# validation_data가 있고, 교차검증을 사용할 때는 pipeline을 하지 않으면 train데이터에 validation데이터가 포함되어서 새로운 데이터로 예측할 때 오차가 많이남

search.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
# print('predict: \n', search.predict(x_test))



model_dic = search.best_params_

model = build_network(model_dic['kcf__keep_prob'], model_dic['kcf__optimizer'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=model_dic['kcf__batch_size'], epochs=500, callbacks=[early_stopping], verbose=0)

print('loss, acc: ', model.evaluate(x_test, y_test))
# loss, acc:  [0.7378770782575979, 0.7597402574180009]