# input_shape=(4,1)
# RMSE, R2(0.95 이상 나오게) 적용

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


# 1. 데이터
x_train = np.array(range(1,97))
y_train = np.array(range(5,101))

print(x_train.shape)    # (96,)

size = 4
# LSTM에 넣기 좋은 연속 데이터 만들기
def split_n(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

x_train = split_n(x_train, size)
y_train = split_n(y_train, size)

print(x_train.shape)    # (93, 4)
print(y_train.shape)    # (93, 4)

# LSTM은 몇 개씩 잘라서 작업할 건가를 지정해야하기에 reshape
x_train = np.reshape(x_train, (x_train.shape[0], 4, 1))

print(x_train.shape)
# print(x_train)

# x_test = np.array(range(97,104))
# y_test = np.array(range(101,108))
x_test = np.array(range(47,54))
y_test = np.array(range(51,58))

print(x_train.shape)    # (96,)

x_test = split_n(x_test, size)
y_test = split_n(y_test, size)

x_test = np.reshape(x_test, (x_test.shape[0], 4, 1))


# 2. 모델 구성
# LSTM 은 층이 하나라도 많은 연산을 함
# return_sequences=True, 다음 계층 LSTM일때 LSTM 데이터로 넘겨주기 위한 옵션
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1)))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
# loss:  0.00897820945829153
# acc:  0.00897820945829153
# RMSE :  0.09475341732590874
# R2 :  0.9928174319240497
# y_predict(x_test) :
#  [[51.114815 52.184074 53.16981  54.11303 ]
#  [52.057404 53.128906 54.122524 55.05545 ]
#  [52.987213 54.0625   55.06184  55.985413]
#  [53.934303 55.014862 56.016346 56.93625 ]]

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
print('acc: ', acc)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))   # 낮을수록 좋음

# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

print('y_predict(x_test) : \n', y_predict)

# import sys
# sys.exit()