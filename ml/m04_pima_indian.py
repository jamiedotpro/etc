from keras.models import Sequential
from keras.layers import Dense
import os
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dir_path = os.getcwd()
pima_indians_file = os.path.join(dir_path, 'etc/data/pima-indians-diabetes.csv')
dataset = numpy.loadtxt(pima_indians_file, delimiter=',')
# dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# 모델의 설정
model = Sequential()
# model.add(Dense(64, input_dim=8, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# Accuracy: 0.9349

model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Accuracy: 0.9753

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=1000, batch_size=5, callbacks=[early_stopping])

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(x, y)[1]))

# sigmoid 결과를 분류로 출력함
# y_predict = model.predict_classes(x)
# print(y_predict)
