from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D
model = Sequential()
model.add(Conv2D(filter_size, kernel_size, input_shape=(28,28,1)))
model.add(Conv2D(16,(3,3)))

model.summary()