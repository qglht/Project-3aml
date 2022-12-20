import numpy as np
import cv2
from typing import Tuple
import random
from scipy import ndimage
import ipdb
from data_treatment import visualize
# def random_crop(img:np.array, yimg:np.array, width:int, height:int)->np.array:
#     assert img.shape[0] >= height
#     assert img.shape[1] >= width
#     x = random.randint(0, img.shape[1] - width)
#     y = random.randint(0, img.shape[0] - height)
#     return img[y:y+height, x:x+width], yimg[y:y+height, x:x+width]

# def shear(img:np.array, sign:int=1)->np.array:
#     shear_factor = sign*random.uniform(0,0.5)
#     if shear_factor < 0:
#         b = -shear_factor * img.shape[1]
#     else:
#         b=0
#     w = img.shape[1]
#     M = np.array([[1, shear_factor, b],[0,1,0]])
#     nW =  img.shape[1] + abs(shear_factor*img.shape[0])
#     img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
#     img = cv2.resize(img, (img.shape[1],img.shape[0]))
#     return np.array([[_ for _ in img[i][int(np.floor((img.shape[1]-w)/2)): int((np.floor(img.shape[1]+w)/2))]] for i in range(img.shape[0])])

# def translate(img: np.array, axis:int=0, shift:int = 20)->np.array:
#     height, width = img.shape
#     rolled = np.roll(img, shift, axis=[1])
#     if axis and shift > 0:
#         rolled = cv2.rectangle(rolled, (0, 0), (shift, height), 0, -1)
#     elif axis:
#         rolled = cv2.rectangle(rolled, (width+shift, 0), (width, height), 0, -1)
#     elif shift > 0:
#         rolled = cv2.rectangle(rolled, (0, 0), (width, shift), 0, -1)
#     else:
#         rolled = cv2.rectangle(rolled, (0, height+shift), (width, height), 0, -1)
#     return rolled

# def augment(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
#     X_augmented, y_augmented = [], []
#     for i, img in enumerate(X):
#         yimg = y[i]
#         for _ in range(2):
#             X_augmented.append(shear(img))
#             y_augmented.append(shear(yimg))
#             X_augmented.append(shear(img, -1))
#             y_augmented.append(shear(yimg, -1))
#         # Image quality
#         X_augmented.append(ndimage.gaussian_filter(img, .5))
#         y_augmented.append(ndimage.gaussian_filter(yimg, .5))
#         X_augmented.append(ndimage.gaussian_filter(img, 1))
#         y_augmented.append(ndimage.gaussian_filter(yimg, 1))
#         # Translation
#         for shift in [10, -10, 20, -20]:
#             X_augmented.append(translate(img, 0, shift))
#             y_augmented.append(translate(yimg, 0, shift))
#             X_augmented.append(translate(img, 1, shift))
#             y_augmented.append(translate(yimg, 1, shift))
#     return np.array(X_augmented), np.array(y_augmented)

from keras.preprocessing.image import ImageDataGenerator



def augment(x_train, y_train : Tuple[np.array, np.array]) -> np.array:
    x_train, y_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1), y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2],1)

    datagen_images = ImageDataGenerator(
    shear_range=20,
    horizontal_flip=False,
    brightness_range=[0.4,1],
    zoom_range=0.3, 
    fill_mode='constant')

    for i in range(x_train.shape[0]):
        x_train_sample = x_train[i,:,:,:]
        y_train_sample = y_train[i,:,:,:]
        for batch_x, batch_y in zip(datagen_images.flow(np.array([x_train_sample]),batch_size=16, seed=1), datagen_images.flow(np.array([y_train_sample]),batch_size=32, seed=1)):
            x_data_aug = batch_x[0,:,:,:]
            y_data_aug = batch_y[0,:,:,:]
            visualize([x_train_sample,y_train_sample,x_data_aug,y_data_aug])







'''
tf.image.stateless_random_brightness
tf.image.stateless_random_contrast
tf.image.stateless_random_crop
tf.image.stateless_random_flip_left_right
tf.image.stateless_random_flip_up_down
tf.image.stateless_random_hue
tf.image.stateless_random_jpeg_quality
tf.image.stateless_random_saturation
'''