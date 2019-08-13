import numpy as np
# 각종 샘플 데이터 셋
# keras ==> numpy
# sklearn ==> data, target

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist_x = np.vstack((np.array(x_train), np.array(x_test)))
mnist_y = np.hstack((np.array(y_train), np.array(y_test)))
np.save('mnist_x.npy', mnist_x)
np.save('mnist_y.npy', mnist_y)
# x = np.load('mnist_x.npy')
# y = np.load('mnist_y.npy')
print('mnist_x', mnist_x.shape)
print('mnist_y', mnist_y.shape)

# x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
# x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
# y_train = y_train.reshape(y_train.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)
# mnist_train = np.hstack((x_train, y_train))
# mnist_test = np.hstack((x_test, y_test))

# mnist_train = np.vstack((mnist_train, mnist_test))
# np.save('mnist_train.npy', mnist_train)
# print(mnist_train.shape)   # (70000, 785)


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar10_x = np.vstack((np.array(x_train), np.array(x_test)))
cifar10_y = np.vstack((np.array(y_train), np.array(y_test)))
np.save('cifar10_x.npy', cifar10_x)
np.save('cifar10_y.npy', cifar10_y)
# x = np.load('cifar10_x.npy')
# y = np.load('cifar10_y.npy')
print('cifar10_x', cifar10_x.shape)
print('cifar10_y', cifar10_y.shape)


from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
boston_housing_x = np.vstack((np.array(x_train), np.array(x_test)))
boston_housing_y = np.hstack((np.array(y_train), np.array(y_test)))
np.save('boston_housing_x.npy', boston_housing_x)
np.save('boston_housing_y.npy', boston_housing_y)
print('boston_housing_x', boston_housing_x.shape)
print('boston_housing_y', boston_housing_y.shape)


# from sklearn.datasets import load_boston
# boston = load_boston()
# print(boston.keys())    # data, target
# boston.data : x값, numpy
# boston.target : y값, numpy

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
label = cancer.target.reshape(-1,1)
print('cancer_data', cancer.data.shape)
print('cancer_target', cancer.target.shape)
# cancer_data = np.c_[cancer.data, label]
cancer_data = np.column_stack((cancer.data, label))

np.save('cancer_data.npy', cancer_data)
cancer_d = np.load('cancer_data.npy')
print('cancer',cancer_d.shape)


import os
dir_path = os.getcwd()
pima_indians_file = os.path.join(dir_path, 'etc/data/pima-indians-diabetes.csv')
dataset = np.loadtxt(pima_indians_file, delimiter=',')
np.save('pima-indians.npy', dataset)
pima = np.load('pima-indians.npy')
print(pima.shape)

wine_file = os.path.join(dir_path, 'etc/data/wine.csv')
dataset = np.loadtxt(wine_file, delimiter=',', encoding='utf-8')
np.save('wine.npy', dataset)
wine = np.load('wine.npy')
print(wine.shape)

# iris_data_file = os.path.join(dir_path, 'etc/data/iris.csv')
# iris_data = np.loadtxt(iris_data_file, delimiter=',')#, encoding='utf-8')
# np.save('iris.npy', iris_data)


def name_class(y):
    for i in range(len(y)):
        if y[i] == b'Iris-setosa':
            y[i] = 0
        elif y[i] == b'Iris-versicolor':
            y[i] = 1
        else:
            y[i] = 2

    return y

import pandas as pd

iris_data_file = os.path.join(dir_path, 'etc/data/iris2.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8')

x = np.array(iris_data.iloc[:, :-1])
y = name_class(iris_data.iloc[:, -1])

y = np.array(y,dtype=np.int32)
iris2_data = np.c_[x, y]
np.save('iris2_data.npy', iris2_data)
# np.save('iris2_label.npy',y)

iris2_data = np.load('iris2_data.npy')
# iris2_label = np.load('iris2_label.npy')

print('iris2_data:',iris2_data.shape)
# print('iris2_label:',iris2_label.shape)