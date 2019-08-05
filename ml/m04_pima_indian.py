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

print(dataset[0])
import sys
sys.exit()


# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(x, y, epochs=200, batch_size=10)

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(x, y)[1]))

# y_predict = numpy.round(model.predict(x))
# print(y_predict)

# import sys
# sys.exit()
