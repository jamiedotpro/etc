from keras.models import Sequential

# filter_size = 32
# kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
# cnn: model.add(Conv2D(...
# filter_size: 입력값 만큼의 이미지 개수를 만들어라(dense의 output 역할)
# kernel_size: 특성을 찾기 위해 이미지를 어떻게 잘라서 작업할 것인가. (3,3): 3 by 3 크기로 잘라서 작업
# input_shape: 입력되는 이미지는 28 by 28. 1은 흑백, 3이 컬러
# model.add(Conv2D(filter_size, kernel_size, input_shape=(28,28,1)))
# padding : 경계 처리 방법을 정의합니다.
    # ‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다. default. (10,10,1)->(9,9,개수)
    # ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.    # shape가 똑같이 나온다.
        # 이미지 바깥에 0을 채워서 이미지 크기를 유지한다.
model.add(Conv2D(7, (2,2), padding='same', input_shape=(10,10,1)))
# model.add(Conv2D(16, (2,2), padding='same'))

# MaxPooling2D 중복없이 자른다. 그 중 가장 특성이 높은 값을 고른다.
    # (2,2) 입력 시 2by2로 자른다. 레이어에 있는 shape를 입력값만큼 변경 가능.
    # input_shape를 MaxPooling2D 입력값으로 자를 때 남는 부분은 버려진다.
model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8, (2,2), padding='valid'))
# model.add(Dropout(0.2))   # Conv2D에도 쓸 수 있음.
model.add(Flatten())    # CNN의 마지막에 추가해서 데이터를 쭉 펴준다.
model.add(Dense(1))     # 출력 하나로 만든다. 앞에 Dense 여러개 추가해도 상관없음.

model.summary()
# (2,2) 사이즈로 잘랐더니 4 by 4 크기가 됨. 16 장을 만듦 (output shape)
# 5 by 5를 (2,2)자르면.. 5-2+1 = 4. 4 by 4가 됨
# 7은 위의 4*4 개수의 이미지를 7장 출력하라 (4*4*7 = 112장 출력됨)
# 첫번째 param: 2*2+1(바이어스) * 7장 = 35
# 두번째... (3,3) 사이즈로 잘랐더니 2 by 2 크기의 16장을 만듦 (output shape)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 4, 4, 7)           35
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 2, 2, 16)          1024
# =================================================================
# Total params: 1,059
# Trainable params: 1,059
# Non-trainable params: 0
# _________________________________________________________________