# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([4, 5, 6])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    # 순차적으로 내려가는 모델이 Sequential

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=200, batch_size=3)
model.fit(x, y, epochs=100)

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=3)
print('acc : ', acc)

y_predict = model.predict(x2)
print(y_predict)