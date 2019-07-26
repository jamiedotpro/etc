# 1. 데이터
import numpy as np

# x = np.array(range(1, 101))
# y = np.array(range(1, 101))
x = np.array([range(1000), range(3110, 4110), range(1000)])
y = np.array([range(5010, 6010)])

print(x.shape)
print(y.shape)


x = np.transpose(x) # 행과 열 바꿈
y = np.transpose(y)
print(x.shape)  # 행렬 확인하기
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

# 2. 모델 구성
from keras.models import load_model
model = load_model('savetest01.h5')


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

from keras.callbacks import EarlyStopping, TensorBoard
# tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=1, callbacks=[early_stopping, tb_hist], validation_data=(x_val, y_val))

# cmd
# cmd 창에서 위에 입력한 경로로 이동: TensorBoard(log_dir='입력한경로'...)
# cmd 창에 우측의 명령어 입력: tensorboard --logdir=./graph
# 크롬에서 http://localhost:6006



# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))

# RMAE : RMSE 와 다르게 절대값으로 계산
#from sklearn.metrics import mean_absolute_error


# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
# 다른 평가모델과 r2를 혼용해서 쓰는 경우가 있음
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)
print('loss : ', loss)