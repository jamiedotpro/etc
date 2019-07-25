from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# y=wx+b
# y = ax^2 + bx + c
# y' = 2ax + b (미분)

# 3차함수 미분
# y = ax^3 + bx^2 + cx + d
# y' = 3ax^2 + 2bx + c
# y'' = 6ax + 2b


# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12]]) # 10행 3열
y = array([4,5,6,7,8,9,10,11,12,13])    # 10행

print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

# reshape(행개수10, 열개수3, 1개씩 자름)
x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape: ', x.shape)


# 2. 모델 구성
# RNN 시계열에 주로 사용. 시간 같이 연속적인 데이터에 잘 맞는다.
# RNN 안에 LSTM(RNN 중 LSTM이 가장 많이 사용됨) 이 포함된다.
# LSTM에서 Param 계산이 어떻게 나오는지 찾기
# 연속된 데이터의 다음 숫자를 찾는 모델
model = Sequential()
# input_shape(컬럼 수, 몇개씩 잘라서 계산할지 개수) 행 개수는 무시되기에 생략함.
# RNN 쓰고 싶으면 아래의 LSTM 부분을 RNN으로 바꾸면 됨. GRU도 마찬가지
# 아래 모델은 LSTM + DNN 임.
# RNN, LSTM 같은 경우 데이터의 최소/최대값을 벗어나면 예측이 잘 안됨.
# DNN은 벗어나도 그럭저럭 맞는 편이라함
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

model.summary()

# LSTM param의 수 :
# input은 input_shape(??, 여기가 input값)
# params = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)
# 480 = 4 * ((1 + 1) * 10 + 10^2)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 10)                480
# _________________________________________________________________
# dense_1 (Dense)              (None, 500)               5500
# _________________________________________________________________
# dense_2 (Dense)              (None, 500)               250500
# _________________________________________________________________
# dense_3 (Dense)              (None, 500)               250500
# _________________________________________________________________
# dense_4 (Dense)              (None, 500)               250500
# _________________________________________________________________
# dense_5 (Dense)              (None, 500)               250500
# _________________________________________________________________
# dense_6 (Dense)              (None, 500)               250500
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 501
# =================================================================
# Total params: 1,258,981
# Trainable params: 1,258,981
# Non-trainable params: 0
# _________________________________________________________________


model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200)

x_input = array([11,12,13])
#x_input = array([70,80,90])
x_input = x_input.reshape((1,3,1))  # 1행 3열에 하나씩 잘라서

yhat = model.predict(x_input)
print(yhat)
