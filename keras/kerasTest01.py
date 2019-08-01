# 2019-08-01 오늘 종가 맞추기
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler

use_colab = False
if use_colab:
  # google.colab 에서 구글 드라이브 파일 로드해서 데이터 읽어오기
  from google.colab import drive
  drive.mount('/content/drive')
  kospi = pd.read_csv('/content/drive/My Drive/Colab Notebooks/kospi200test.csv', encoding='CP949')
else:
  # 1. 데이터 읽어오기
  dir_path = os.path.dirname(os.path.abspath(__file__))
  kospi200_file = os.path.join(dir_path, 'kospi200test.csv')
  kospi = pd.read_csv(kospi200_file, encoding='CP949')

print(kospi.head())

# 일자,         시가,       고가,       저가,       종가,       거래량, 환율(원/달러),
# 2019/07/31, 2036.46,    2041.16,    2010.95,    2024.55,    589386, 1183.1,
# 2019/07/30, 2035.32,    2044.59,    2032.61,    2038.68,    547029, 1181.6,
# 2019/07/29, 2059.13,    2063.13,    2025.01,    2029.48,    608670, 1183.5,
# 2019/07/26, 2063.35,    2068.16,    2054.64,    2066.26,    589074, 1184.8,
# 2019/07/25, 2085.67,    2088.81,    2061.08,    2074.48,    598634, 1181.5,

# 2017/03/13,   2102.37,    2122.88,    2100.91,    2117.59,    382129, 1144.4,
# 2017/03/10,   2088.67,    2102.05,    2082.31,    2097.35,    500860, 1157.4,
# 2017/03/09,   2098.29,    2100.08,    2090.73,    2091.06,    408229, 1158.1,
# 2017/03/08,   2092.27,    2101.56,    2088.19,    2095.41,    381441, 1145.5,
# 2017/03/07,   2080.77,    2096.79,    2079.16,    2094.05,    277830, 1146.1,
# 2017/03/06,   2073.2,     2083.8,     2067.68,    2081.36,    431445, 1158,
# 2017/03/03,   2090.92,    2091.59,    2072.09,    2078.75,    425935, 1156.1,
# 2017/03/02,   2105.19,    2112.58,    2096.31,    2102.65,    434713, 1141.6,
# 2017/02/28,   2087.26,    2094.41,    2084.36,    2091.64,    399821, 1130.7,
# 2017/02/27,   2095.47,    2097.24,    2084.08,    2085.52,    343702, 1133.7,
# 2017/02/24,   2106.43,    2107.83,    2090.05,    2094.12,    385394, 1131.5,
# 2017/02/23,   2106.15,    2108.99,    2103.11,    2107.63,    430850, 1137.3,
# 2017/02/22,   2106.42,    2108.98,    2101.56,    2106.61,    312219, 1142.6,
# 2017/02/21,   2085.97,    2108.48,    2085,       2102.93,    292539, 1146.1,
# 2017/02/20,   2084.16,    2085.59,    2077.13,    2084.39,    297203, 1147.5,
# 2017/02/17,   2072.57,    2081.18,    2072.57,    2080.58,    301517, 1146.3,

# 2117.59
# 2097.35
# 2091.06
# 2095.41
# 2094.05
# 2081.36
# 2078.75
# 2102.65
# 2091.64
# 2085.52
# 2094.12
# 2107.63
# 2106.61
# 2102.93
# 2084.39
# 2080.58

# date = np.array(kospi['일자'])
last_price = np.array(kospi['종가'])
# print('type:', type(last_price))
# print(last_price)

size = 5
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(last_price, size)
print('===================================')
print(dataset)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4:]

print(x_train.shape)    # (595, 4)
print(y_train.shape)    # (595, 1)


# 원활한 훈련을 위해 훈련 데이터 스케일 조정
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)

# LSTM은 몇개씩 잘라서 작업할 것인가를 정해야함
x_train = np.reshape(x_train, (last_price.shape[0]-size+1, 4, 1))
print(x_train.shape)    # (595, 4, 1)

# 테스트할 데이터는 20일치를 가져옴
test_days = 20
last_price_last_days = last_price[last_price.shape[0]-test_days:]
dataset_test = split_5(last_price_last_days, size)
x_test = dataset_test[:, 0:4]
y_test = dataset_test[:, 4:]

print(x_test)
print(y_test)
# x_test = np.append(x_test, [2106.61, 2102.93, 2084.39, 2080.58])
# y_test = np.appnd(y_test, [2080.58])
print(x_test)
print(y_test)

x_test = scaler.transform(x_test)
x_test = np.reshape(x_test, (test_days-size+1, 4, 1))

# 2. 모델 구성
model = Sequential()

# model.add(LSTM(128, input_shape=(4, 1), return_sequences=True))
# model.add(LSTM(32))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='relu'))
# loss:  140.95263671875
# acc:  140.95263671875

# model.add(LSTM(128, input_shape=(4, 1), return_sequences=True))
# model.add(LSTM(32, return_sequences=True))
# model.add(LSTM(4))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='relu'))
# loss:  152.17245483398438
# acc:  152.17245483398438
# RMSE :  12.335851243863786
# R2 :  -0.3404910789915949

model.add(LSTM(128, input_shape=(4, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))


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