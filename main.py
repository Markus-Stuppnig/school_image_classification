import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
