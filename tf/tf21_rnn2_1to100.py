# 1 ~ 100 까지의 숫자를 이용해서 6개씩 잘라서 rnn 구성
# train, test 분리할 것

# 1,2,3,4,5,6 : 7
# 2,3,4,5,6,7 : 8
# 3,4,5,6,7,8 : 9
# ...
# 94,95,96,97,98,99 : 100

# predict: 101 ~ 110까지 예측하시오.
# 지표 RMSE

import numpy as np

def split_n(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

data = np.array([i for i in range(1, 101)])
dataset = split_n(data, 7)

x = dataset[:, :6]
y = dataset[:, 6:]

# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

x_train = x_train.reshape(-1, 6, 1)
x_test = x_test.reshape(-1, 6, 1)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(32, input_shape=(6, 1), return_sequences=True))
model.add(LSTM(4))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=1, callbacks=[early_stopping])

# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
# print('acc: ', acc)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))   # 낮을수록 좋음

test_data = np.array([i for i in range(101, 111)])
test_dataset = split_n(test_data, 6)
# print(test_dataset)
test_dataset = scaler.transform(test_dataset)
test_dataset = test_dataset.reshape(-1, 6, 1)
y_predict = model.predict(test_dataset)
print(y_predict)