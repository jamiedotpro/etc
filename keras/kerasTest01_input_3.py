# 2019-08-01 오늘 종가 예측하기
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 1. 데이터 읽어오기

use_colab = False
if use_colab:
  # google.colab 에서 구글 드라이브 파일 로드해서 데이터 읽어오기
  from google.colab import drive
  drive.mount('/content/drive')
  kospi = pd.read_csv('/content/drive/My Drive/Colab Notebooks/kospi200test.csv', encoding='CP949')
else:
  # 컴퓨터 안의 데이터 읽어오기
  dir_path = os.path.dirname(os.path.abspath(__file__))
  kospi200_file = os.path.join(dir_path, 'kospi200test.csv')
  kospi = pd.read_csv(kospi200_file, encoding='CP949')

# print(kospi.head())

# 일자,         시가,       고가,       저가,       종가,       거래량, 환율(원/달러),
# 2019/07/31, 2036.46,    2041.16,    2010.95,    2024.55,    589386, 1183.1,
# 2019/07/30, 2035.32,    2044.59,    2032.61,    2038.68,    547029, 1181.6,
# 2019/07/29, 2059.13,    2063.13,    2025.01,    2029.48,    608670, 1183.5,
# 2019/07/26, 2063.35,    2068.16,    2054.64,    2066.26,    589074, 1184.8,
# 2019/07/25, 2085.67,    2088.81,    2061.08,    2074.48,    598634, 1181.5,

# 2017/02/23,   2106.15,    2108.99,    2103.11,    2107.63,    430850, 1137.3,
# 2017/02/22,   2106.42,    2108.98,    2101.56,    2106.61,    312219, 1142.6,
# 2017/02/21,   2085.97,    2108.48,    2085,       2102.93,    292539, 1146.1,
# 2017/02/20,   2084.16,    2085.59,    2077.13,    2084.39,    297203, 1147.5,
# 2017/02/17,   2072.57,    2081.18,    2072.57,    2080.58,    301517, 1146.3,

high_price = np.array(kospi['고가'])
low_price = np.array(kospi['저가'])
last_price = np.array(kospi['종가'])

high_price = np.reshape(high_price, (-1, 1))
low_price = np.reshape(low_price, (-1, 1))
last_price = np.reshape(last_price, (-1, 1))


# 원활한 훈련을 위해 훈련 데이터 스케일 조정
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(last_price)
# high_price = scaler.transform(high_price)
# low_price = scaler.transform(low_price)
# last_price = scaler.transform(last_price)

train_data = np.concatenate([high_price, low_price, last_price], axis=1)

size = 6
def split_n(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_n(train_data, size)
print('===================================')


x_train = dataset[:, :dataset.shape[1]-1]
y_train = dataset[:, dataset.shape[1]-1:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2], 1))


# 종가 데이터만 가져오기
y_train = y_train[:, :, -1]


# 테스트할 데이터는 2개 결과를 볼 수 있게 작업
test_days = size * 2 - 1
last_price_last_days = train_data[-test_days:]

# 내일 데이터 예측을 위해 테스트 데이터 하나 더 추가
test_days += 1
last_price_last_days = np.append(last_price_last_days, train_data[-1:], axis=0)

dataset_test = split_n(last_price_last_days, size)

x_test = dataset_test[:, :dataset_test.shape[1]-1]
y_test = dataset_test[:, dataset_test.shape[1]-1:]


# x_test = scaler.transform(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2], 1))


# 종가 데이터만 가져오기
y_test = y_test[:, :, -1]


# import sys
# sys.exit()


# 2. 모델 구성
model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(LSTM(8))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='relu'))


model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adadelta', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, callbacks=[early_stopping])


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
