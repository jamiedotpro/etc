from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)
print(Y_test.shape)


# 하이퍼 파라미터 최적화
# CNN 모델로 만들기
# 계산된 최적의 값으로 CNN 모델링 하기
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden1')(inputs)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', name='hidden2')(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', name='hidden3')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [5,10,20]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용

search.fit(X_train, Y_train)

print(search.best_params_)

# [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:  8.1min finished
# Epoch 1/1
# 60000/60000 [==============================] - 24s 392us/step - loss: 0.1125 - acc: 0.9651
# {'optimizer': 'adadelta', 'keep_prob': 0.1, 'batch_size': 20}

# [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed: 20.7min finished
# Epoch 1/1
# 60000/60000 [==============================] - 48s 800us/step - loss: 0.1253 - acc: 0.9625
# {'optimizer': 'adadelta', 'keep_prob': 0.2, 'batch_size': 10}