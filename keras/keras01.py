# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu')) # input은 1개로. output 노드 5개. input layer/hidden layer
model.add(Dense(3)) # output 노드 3개. hidden layer
model.add(Dense(4)) # output 노드 4개. hidden layer
model.add(Dense(1)) # output 노드 1개. output layer
# hidden layer: 컴퓨터가 스스로 훈련하고 학습하는 부분

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)   # model.fit 훈련시키다 # epochs=반복학습을 할 횟수 # batch_size=몇개씩 잘라서 학습을 할 것인가

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('acc : ', acc)


# 선형회귀 == 회귀분석 == 회귀모델 == y=wx+b
# 딥러닝은 1차함수로 이루어짐. 딥러닝은 최적의 weight 값을 찾는 것
# weight, bias. 최적의 weight 값을 구해야함. bias는 영향이 크지 않음
# y=wx+b 를 얼마나 깔끔하게 만드는가가 핵심
# y' = wx' 새로운 데이터 x'를 받아서 y'를 예측하는 것
# 데이터 전처리가 90%