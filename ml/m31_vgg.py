from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False,
                    input_shape=(150,150,3))
# conv_base = VGG16()

conv_base.summary()

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 다운로드
# from keras.applications import Xception, InceptionV3, ResNet50, VGG19, MobileNet
# conv_base = Xception()
# conv_base = InceptionV3()
# conv_base = ResNet50()
# conv_base = VGG19()
# conv_base = MobileNet()
