# 1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = x[0:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[0:60]
y_val = y[60:80]
y_test = y[80:]


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    # 순차적으로 내려가는 모델이 Sequential

# 깊이, 노드 개수, epochs, batch_size 순으로 조정
model.add(Dense(5, input_dim=1, activation='relu')) # input_dim. 입력되는 데이터의 열개수
#model.add(Dense(10, input_shape=(1, ), activation='relu'))  # input_shape = (1, ) ==> 행은 모르고, 열이 하나
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))
# 대회, 레이어 20개 전후라고 함

# 3. 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))   # validation_data 훈련하면서 검증까지 함. 추가하면 훈련이 더 잘됨
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