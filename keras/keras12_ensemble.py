# 1. 데이터
import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])
x2 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size=0.4)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=66, test_size=0.5)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size=0.4)
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=66, test_size=0.5)


# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()   # 순차적인 모델

# 함수형 모델
# 앙상블의 상위 2개 입력층 작업만 함
input1 = Input(shape=(3, )) # shape에 데이터 컬럼 수
dense1 = Dense(100, activation='relu')(input1) # 괄호 안에 입력데이터를 넣어준다.
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3, ))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])   # 모델 합치기

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)

################################# 
# 여기부터 아웃풋 모델

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(70)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs = [input1, input2], outputs = [output1_3, output2_3])

model.summary()


# 3. 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=1, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y1_val]))   # validation_data 훈련하면서 검증까지 함. 추가하면 훈련이 더 잘됨
#model.fit(x_train, y_train, epochs=500)


# 4. 평가 예측
# accuracy 분류모델.
# mse: mean_squared_error 평균제곱오차
# rmse : sqrt(mse)
acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
#print('model.metrics_names', model.metrics_names)
#model.metrics_names ['loss', 'dense_11_loss', 'dense_14_loss', 'dense_11_mean_squared_error', 'dense_14_mean_squared_error']
                        # total_loss, model1_loss, model2_loss, acc1, acc2
print('acc: ', acc)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict)
print(y2_predict)


# 루트 씌우는 이유는 숫자가 커서 줄이려고. 다른 이유는 없다함
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print('RMSE1 : ', RMSE1)   # 낮을수록 좋음
print('RMSE2 : ', RMSE2)
print('RMSE : ', (RMSE1 + RMSE2)/2)

# RMAE : RMSE 와 다르게 절대값으로 계산
#from sklearn.metrics import mean_absolute_error


# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
# 다른 평가모델과 r2를 혼용해서 쓰는 경우가 있음
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print('R2 _1 : ', r2_y1_predict)
print('R2 _2 : ', r2_y2_predict)
print('R2 : ', (r2_y1_predict + r2_y2_predict) / 2)