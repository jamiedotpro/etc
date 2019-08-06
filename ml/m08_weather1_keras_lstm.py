import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np


# 기온 데이터 읽어 들이기
dir_path = os.getcwd()
tem10y_file = os.path.join(dir_path, 'etc/data/tem10y.csv')
df = pd.read_csv(tem10y_file, encoding='utf-8')


# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data, interval=6):
    x = []  # 학습 데이터
    y = []  # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval:
            continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)


x, y = make_data(df)

x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)
print(x_test.shape)


# 원활한 훈련을 위해 훈련 데이터 스케일 조정
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# 학습하기
model = Sequential()

# model.add(LSTM(32, input_shape=(6, 1), activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='relu'))
# loss:  3.681307488720711
# acc:  3.681307488720711
# RMSE :  1.9186733623366525
# R2 :  0.9428506116743651

model.add(LSTM(50, input_shape=(6, 1), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='relu'))
# loss:  3.696310070465095
# acc:  3.696310070465095
# RMSE :  1.9225789822682176
# R2 :  0.942617710160758

model.summary()


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # 회귀

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1,
            validation_split=0.2, callbacks=[early_stopping])


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


# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음. 회귀
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

# print('y_predict(x_test) : \n', y_predict)

# 결과를 그래프로 그리기
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(y_test, c='r')
plt.plot(y_predict, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()
