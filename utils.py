import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam

def preprocess_img(x):
    y = x/127.5 - np.ones(x.shape)
    return y

def low_res(imgs, downscale):
    images = []
    for img in imgs:
        img_shape = img.shape
        img2 = cv2.resize(img, (img_shape[0]//downscale, img_shape[1]//downscale))
        images.append(img2)

    images = np.array(images)
    return images

def load_images(path):
    images= []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))

        if len(img.shape)== 3:
            img = cv2.resize(img, (128,128))
            images.append(img)


    train_size = int(0.8*len(images))
    train_hr = images[:train_size]
    test_hr = images[train_size:]

    train_lr = low_res(train_hr, 4)
    test_lr  = low_res(test_hr, 4)


    train_hr = np.array(train_hr)
    test_hr = np.array(test_hr)

    return train_hr, train_lr, test_hr, test_lr



def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        make_trainable(vgg19, False)
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))
    
def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

def plot_sr(generator, test_hr, test_lr):
    rand_nums = np.random.randint(0, test_lr.shape[0], size = 5)
    test_images = test_lr[rand_nums]
    actual_images = test_hr[rand_nums]
    images = generator.predict(test_images)
    # test_images = np.multiply(np.add(test_images,1.0),255.0/2.0)
    # images = np.multiply(np.add(images,1.0),255.0/2.0)
    plt.figure(figsize = (20,8))
    for i in range(5):
        ax = plt.subplot(5,3,3*i+1)
        plt.imshow(test_images[i])
        ax.set_title('Low Resolution')
        ax = plt.subplot(5,3,3*i+2)
        plt.imshow(images[i])
        ax.set_title('High Resolution')
        ax = plt.subplot(5,3,3*i+3)
        plt.imshow(actual_images[i])
        ax.set_title('Actual Image')
    plt.show()


