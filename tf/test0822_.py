import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, BatchNormalization, Dropout

# 3시간마다 변경된 데이터
# 14년치 데이터 1999-01-01 ~ 2013-12-31
# 2007-07-11 ~  2007-07-15  ( 3113 ~ 3117 번째 데이터 )
# 빈셀 채우기
# 빈 5일치 데이터를 보내면 됨

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test_file = os.path.join(dir_path, 'data/test0822.csv')
test_file = 'd:/test.csv'
test_data = pd.read_csv(test_file)

# print(test_data.head())

# 0 ~ 9 까지의 데이터가 입력되어 있음
# print(test_data['kp_0h'].max())
# print(test_data['kp_3h'].max())
# print(test_data['kp_6h'].max())
# print(test_data['kp_9h'].max())
# print(test_data['kp_12h'].max())
# print(test_data['kp_15h'].max())
# print(test_data['kp_18h'].max())
# print(test_data['kp_21h'].max())

#             date  kp_0h  kp_3h  kp_6h  kp_9h  kp_12h  kp_15h  kp_18h  kp_21h
# 0     1999-01-01    0.0    2.0    1.0    2.0     2.0     1.0     1.0     1.0
# 1     1999-01-02    1.0    2.0    2.0    3.0     3.0     2.0     2.0     1.0
# 2     1999-01-03    2.0    2.0    0.0    0.0     1.0     1.0     1.0     1.0
# 3     1999-01-04    1.0    2.0    3.0    2.0     3.0     2.0     1.0     2.0
# 4     1999-01-05    3.0    3.0    2.0    3.0     1.0     1.0     2.0     1.0

# 3111  2007-07-09    1.0    1.0    0.0    0.0     1.0     1.0     1.0     1.0
# 3112  2007-07-10    1.0    0.0    0.0    1.0     1.0     2.0     1.0     3.0
# 3113  2007-07-11    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
# 3114  2007-07-12    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
# 3115  2007-07-13    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
# 3116  2007-07-14    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
# 3117  2007-07-15    NaN    NaN    NaN    NaN     NaN     NaN     NaN     NaN
# 3118  2007-07-16    1.0    2.0    2.0    1.0     2.0     1.0     1.0     2.0
# 3119  2007-07-17    1.0    1.0    1.0    1.0     1.0     2.0     2.0     2.0

# date = np.array(test_data['date'])
# print(date)

# numpy 데이터 다 보여주도록 옵션 설정
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

kp = np.array(test_data[['kp_0h', 'kp_3h', 'kp_6h', 'kp_9h', 'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h']], dtype=np.int)
# print(kp)
# print(kp.shape)


# 빈 데이터 앞쪽의 데이터
kp_data = kp[:3113, :]

# 데이터 역순으로 변경
def np_reverse(npdata):
    arr = []
    for i in range(-1, -len(npdata)-1, -1):
        arr.append(npdata[i])
    return np.array(arr)


# 빈 데이터 뒤쪽의 데이터
# 역순으로 값을 탐색해야해서 배열을 역순으로 변경
kp_data_reverse = kp[3118:, :]
kp_data_reverse = np_reverse(kp_data_reverse)

size = 30
in_size = 25
out_size = size - in_size
def split_n(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_n(kp_data, size)
dataset_r = split_n(kp_data_reverse, size)


# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

def model_(dataset):

    x = dataset[:, :in_size]
    y = dataset[:, in_size:]

    print(x.shape)  # (3084, 25, 8)
    x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))   # (3084, 200)
    scaler.fit(x)
    x = scaler.transform(x)
    x = np.reshape(x, (x.shape[0], 25, 8))
    print(x.shape)  # (3084, 200)

    # x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], 1))
    # x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:]), 1))
    y = np.reshape(y, (y.shape[0], np.prod(y.shape[1:])))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2, shuffle=True)

    model = Sequential()

    # model.add(LSTM(8, input_shape=(x_train.shape[1], 8), return_sequences=True))
    # model.add(LSTM(4))
    # model.add(Dense(10, activation='relu'))

    model.add(LSTM(32, input_shape=(x_train.shape[1], 8), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(64, activation='relu'))
    # 제출2: 1.5083

    # model.add(LSTM(40, input_shape=(x_train.shape[1], 8), return_sequences=True))
    # model.add(LSTM(20))
    # model.add(Dense(80, activation='relu'))


    model.add(Dense(y_train.shape[1], activation='relu'))

    model.summary()


    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping], validation_split=0.2)


    loss, acc = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    print('loss: ', loss)
    # print('acc: ', acc)

    # 반올림 및 정수로 형변환
    y_predict = np.around(y_predict)
    y_predict = y_predict.astype('int')

    return model, y_test, y_predict


# RMSE 구하기
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


model1, y_test1, y_predict1 = model_(dataset)
model2, y_test2, y_predict2 = model_(dataset_r)

# RMSE(낮을수록 좋음), R2(1에 가까울수록 좋음) 구하기
print()
print('model_predict1, rmse: ', RMSE(y_test1, y_predict1), 'r2: ', r2_score(y_test1, y_predict1))
print('model_predict2, rmse: ', RMSE(y_test2, y_predict2), 'r2: ', r2_score(y_test2, y_predict2))


def cross_check(model1, model2, num1=3113):
    # 3113 ~ 3117 이 빈 데이터
    num2 = num1 + 5
    y_test = kp[num1:num2, :]
    test1 = kp[num1-25:num1, :]
    test2 = kp[num2:num2+25, :]

    print(test1.shape)  # (25, 8)
    test1 = np.reshape(test1, (-1, np.prod(test1.shape)))
    print(test1.shape)
    test2 = np.reshape(test2, (-1, np.prod(test2.shape)))
    test1 = scaler.transform(test1)
    test2 = scaler.transform(test2)
    test1 = np.reshape(test1, (25, 8))
    test2 = np.reshape(test2, (25, 8))
    print(test1.shape)

    test1 = np.reshape(test1, (-1, 25, 8))
    
    # 뒤쪽 데이터는 역순 처리
    test2 = np_reverse(test2)
    test2 = np.reshape(test2, (-1, 25, 8))

    y_test = np.reshape(y_test, (-1, np.prod(y_test.shape)))

    y_predict1 = np.around(model1.predict(test1))
    y_predict2 = np.around(model2.predict(test2))

    if num1 != 3113:
        # RMSE(낮을수록 좋음), R2(1에 가까울수록 좋음) 구하기
        print('predict1, rmse: ', RMSE(y_test, y_predict1), 'r2: ', r2_score(y_test, y_predict1))
        print('predict2, rmse: ', RMSE(y_test, y_predict2), 'r2: ', r2_score(y_test, y_predict2))

    # reshape 및 뒤쪽 데이터 원 순서로 변경
    y_predict1 = np.reshape(y_predict1, (-1, 8))
    y_predict2 = np.reshape(y_predict2, (-1, 8))
    y_predict2 = np_reverse(y_predict2)   # 날짜별 역순으로 넣은 데이터이므로, 원래 순서로 변경

    print('model1_result: ')
    print(y_predict1)
    print('model2_result: ')
    print(y_predict2)

    y_predict3 = np.around((y_predict1 + y_predict2) / 2)
    y_predict3 = y_predict3.astype('int')
    print('model3_result: ')
    print(y_predict3)
    print(type(y_predict3))

    if num1 == 3113:
        np.savetxt('result.csv', y_predict3, delimiter=',', fmt='%i')
    else:
        y_predict3 = np.reshape(y_test, (-1, np.prod(y_test.shape)))
        print('predict3, rmse: ', RMSE(y_test, y_predict3), 'r2: ', r2_score(y_test, y_predict3))


print('\n검증용 데이터: ')
cross_check(model1, model2, 700)

print('\n제출할 데이터: ')
cross_check(model1, model2)



#-----------------------------------------------------
# # 원 데이터. 테스트용
# test_data = np.array([i for i in range(1, 41)])
# test_data = np.reshape(test_data, (5, 8))
# print(test_data)

# # 날짜 기준 역순 처리
# test_data = np_reverse(test_data)
# print(test_data)

# # 모델에 돌릴 수 있도록 reshape
# test_data = np.reshape(test_data, (-1, 40))
# print(test_data)

# #-----------------------------------
# # 모델에서 받은 결과를 reshape
# test_data = np.reshape(test_data, (5, 8))
# print(test_data)

# # 날짜 기준 역순서인 데이터를 받아서 원래 순서로 변경
# test_data = np_reverse(test_data)
# print(test_data)
#-----------------------------------------------------
