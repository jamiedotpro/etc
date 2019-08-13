import numpy as np
a = np.arange(10)
print(a)
np.save('aaa.npy', a)
b = np.load('aaa.npy')
print(b)

# dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# iris_data = pd.read_csv('iris.csv', encoding='utf-8')
#     # index_col = 0, encoding='cp949', sep=',', header=None
#     # names=['x1', 'x2', 'x3', 'x4', y']
# wine = pd.read_csv('wine.csv', sep=';', encoding='utf-8')

#### utf-8 ####
#-*- coding: utf-8 -*-

### pandas 를 numpy로 바꾸기
# 판다스.value

