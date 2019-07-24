# 행은 무시되고, 열이 우선이다.


# 1. 데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x3 = np.array([101, 102, 103, 104, 105, 106])
#x4 = np.array([301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315])
x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    # 순차적으로 내려가는 모델이 Sequential

# 깊이, 노드 개수, epochs, batch_size 순으로 조정
#model.add(Dense(5, input_dim=1, activation='relu')) # input_dim. 입력되는 데이터의 열개수
model.add(Dense(1000, input_shape=(1, ), activation='relu'))  # input_shape = (1, ) ==> 행은 모르고, 열이 하나
model.add(Dense(2249))
model.add(Dense(2356))
model.add(Dense(2566))
model.add(Dense(2345))
model.add(Dense(2456))
model.add(Dense(2516))
model.add(Dense(2300))
model.add(Dense(1))
# 대회, 레이어 20개 전후라고 함

# 3. 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
#model.fit(x_train, y_train, epochs=500)

model.summary()

# 4. 평가 예측
# accuracy 분류모델.
# mse: mean_squared_error 평균제곱오차
# rmse : sqrt(mse)
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

# 루트 씌우는 이유는 숫자가 커서 줄이려고. 다른 이유는 없다함
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))   # 낮을수록 좋음

# RMAE : RMSE 와 다르게 절대값으로 계산
#from sklearn.metrics import mean_absolute_error


# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
# 다른 평가모델과 r2를 혼용해서 쓰는 경우가 있음
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)