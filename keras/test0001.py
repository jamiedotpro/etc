import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
from keras import backend

print('----------------------------------')
print('tensorflow version: ', tf.__version__)
print('keras version: ', keras.__version__)

# gpu 인식하면 모두 True 출력됨
print('----------------------------------')
print('GPU 사용 중인가?: ', tf.test.is_built_with_cuda())
print('GPU 사용 중인가?: ', tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


# 텐서플로가 인식하는 디바이스 중에 GPU 가 있는지 확인
print('----------------------------------')
print(device_lib.list_local_devices())

# 혹은 다음과 같이 인식하는 GPU가 있는지 확인
print('----------------------------------')
print(backend.tensorflow_backend._get_available_gpus())
