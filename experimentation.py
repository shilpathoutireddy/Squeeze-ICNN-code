import numpy as np
from matplotlib import pyplot as plt
import cv2,os
from skimage import color
rn=0
import nibabel as nib,cv2,re
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import jaccard_score,accuracy_score
import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
db=1
if db == 1:
    # read image #
    DB_path = ['Database//archive 3//volume_pt1//']
import os,cv2
from skimage.transform import resize
import numpy as np,glob,pickle
import warnings,sys,os,pickle,random
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
def pred_(fp,p):
    if 'rmal' in fp: return 0
    elif 'non'in fp: return 0
    else: return 1
from skimage.transform import resize
import numpy as np,cv2,glob,matplotlib.pyplot as plt,skimage



def predict(mdl,image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret4, th4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = cv2.bitwise_not(th4)

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # color_ch0 = hsv[:, :, 2]  # first channel
    # seg=color_ch0 < 200
    # plt.imshow(color_ch0 < 200)
    # thr_1 = 100  # min and max value
    # thr_2 = 255
    # seg = (color_ch0 > thr_1) & (color_ch0 < thr_2)
    # seg = seg.astype('uint8')
    plt.imshow(seg,cmap='gray')

    return seg


def cunet(gray):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype('uint8')
    ret4, th4 = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = cv2.bitwise_not(th4)
    return seg

def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    import math
    def chebyshev_map():
        ck = np.random.uniform(0, 1)
        ck_1 = math.cos(0.5 * math.acos(ck))
        return ck_1

    # t1, t2=try, except
    import keras.backend as K
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)
        y_true = K.switch(K.greater(smoothing, 0),
                          lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                          lambda: y_true)
    M = y_pred.shape[1].value
    P = K.sum(y_pred==1)
    N = K.sum(y_pred==0)
    if y_true==1:
        Bi = (P + N)/P
    else:
        Bi = (P + N) / N
    Ubce = (1/M) * K.sum(Bi)
    # Wi = chebyshev_map()
    # updated equation #
    ls = K.mean(- (1 / M) * K.sum(Ubce * y_true * K.log(y_pred)) + ((1 - y_true) * K.log(1 - y_pred)))  # improved
    return ls


def train_UNetmodel():
    ############################## X Train #############################


    path = 'Database//ALL_IDB1//ALL_IDB1//im*.jpg'  ### DB
    all_file = glob.glob(path)

    input_file, gt_file = [], []  # normal
    for kk in range(len(all_file)):
        if 'exp' not in all_file[kk]:
            input_file.append(all_file[kk])
        elif 'exp1' in all_file[kk]:
            gt_file.append(all_file[kk])

    path1= 'Database//RIM_ONE_database_r1//glaucoma/*.bmp'  ### DB
    all_file1 = glob.glob(path1)
    input_file1, gt_file1 = [], []  # glaucoma
    for kk in range(len(all_file1)):
        if 'exp' not in all_file1[kk]:
            input_file1.append(all_file1[kk])
        elif 'exp1' in all_file1[kk]:
            gt_file1.append(all_file1[kk])

    files=np.concatenate((input_file,input_file1))
    X_train = np.zeros((len(files), 128, 128, 1), dtype=np.uint8)
    Y_train = np.zeros((len(files), 128, 128, 1), dtype=np.bool)
    for i in range(len(files)):

        # read the image using skimage
        image = cv2.imread(files[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize the image
        image = resize(image, (128, 128), mode='constant', preserve_range=True)
        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        image = np.expand_dims(image, axis=-1)
        # insert the image into X_train
        X_train[i] = image
    print(X_train.shape)

    ################################# Y train ###############################

    gt_files =np.concatenate((gt_file,gt_file1))
    for i in range(len(gt_files)):

        # read the image using skimage
        image = cv2.imread(gt_files[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize the image
        image = resize(image, (128, 128), mode='constant', preserve_range=True)
        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        image = np.expand_dims(image, axis=-1)
        # insert the image into X_train
        Y_train[i] = image
    print(Y_train.shape)

    ################################################################################
    from keras.models import Model, load_model
    from keras.layers import Input
    from keras.layers.core import Dropout, Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras import backend as K
    import tensorflow as tf

    # source: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277


    inputs = Input((128, 128, 1))
    net=0
    s = Lambda(lambda x: x / 255) (inputs)
    ######## updated: relu and sigmoid #########

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)  ############# sigmoid activation used

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.summary()

    filepath = "model.h5"

    earlystopper = EarlyStopping(patience=5, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=callbacks_list)

    if net == 1:
        filename = 'model1.sav'
        pickle.dump(model, open(filename, 'wb'))
        model.save_weights('saved data/unetmodel.h5')


def tst_unet_segmentation(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (128, 128))
    kernel = np.ones((60, 60), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ch1 = hsv[:, :, 1]
    IM=im.reshape(1,im.shape[0],im.shape[1],1)
    IMG_HEIGHT = im.shape[0]
    IMG_WIDTH = im.shape[1]
    IMG_CHANNELS = 1
    def img_(im):
        kernel_size = (20, 20)  # should roughly have the size of the elements you want to remove
        kernel_el = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        eroded = cv2.erode(im, kernel_el, (-1, -1))
        cleaned = cv2.dilate(eroded, kernel_el, (-1, -1))
        return cleaned
    image_id_list,NUM_TEST_IMAGES=1,1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ch1 = hsv[:, :, 1]
    unet_net = ch1.copy()
    X_train = np.ones((10, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    X_train=np.concatenate((X_train, IM))
    Y_train = np.ones(((10), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    X_test = np.ones((NUM_TEST_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    model = pickle.load(open('saved model/model1.sav', 'rb'))
    test_preds = model.predict(X_test)
    # Threshold the predictions
    preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)
    unet_ot=preds_test_thresh[0,:,:,:]
    unet_out = predict(model, img)

    return unet_out

def conv_RCNN(im, im_):  ######### proposed
    # import torch
    # from torchsummary import summary
    # import graphviz
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='cm_RCNN.jpg', show_shapes=True, dpi=300)

    def mask_head(feature_map, num_classes):
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(feature_map)
        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)
        return x

    def mask_rcnn(input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)

        # Backbone network (you can use any pre-trained backbone like ResNet, etc.)
        backbone = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        backbone = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.MaxPooling2D(pool_size=(2, 2))(backbone)

        # Add more layers based on the backbone architecture
        backbone = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(backbone)

        # Region Proposal Network (RPN)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(backbone)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(rpn)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(rpn)

        # Region Proposal Network Output
        rpn_class = layers.Conv2D(2, (1, 1), activation='softmax')(rpn)
        rpn_bbox = layers.Conv2D(4, (1, 1))(rpn)

        # Mask Prediction Head
        mask_pred = mask_head(rpn, num_classes)

        model = models.Model(inputs=inputs, outputs=[rpn_class, rpn_bbox, mask_pred])
        return model
    vv = np.load('saved data/gt.npy')

    # Usage
    num_classes = 1  # Number of classes, e.g., tumor
    input_shape = (im.shape[0], im.shape[1], 1)  # Input shape of your grayscale images
    model = mask_rcnn(input_shape, num_classes)

    test_image = im
    test_image = cv2.resize(test_image, (im.shape[0], im.shape[1]))
    test_image = np.reshape(test_image, [1, im.shape[0], im.shape[1]])
    # # test_image = cv2.imread("path/to/image.jpg")
    # test_image = test_image / 255.0  # Normalize to [0, 1]
    # test_image = np.expand_dims(test_image, axis=0)

    # Perform segmentation
    rpn_class, rpn_bbox, mask_pred = model.predict(test_image)
    # mask_pred = (mask_pred * 255).astype('uint8')
    mask_pred = cv2.resize(mask_pred[0, :, :, 0], (im.shape[0], im.shape[1]))
    pred_im = vv * (mask_pred * 255).astype('uint8')
    # cv2.imshow('pred_im', pred_im)
    pred_im = pred_im * im
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.dilate(pred_im, kernel, iterations=1)
    return out


def prop_cm_RCNN(im, im_):  ######### proposed
    # import torch
    # from torchsummary import summary
    # import graphviz
    # from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='cm_RCNN.jpg', show_shapes=True, dpi=300)

    def mask_head(feature_map, num_classes):
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(feature_map)
        x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)

        def sigmoid(x):
            from tensorflow.python.ops import nn
            import tensorflow.keras.backend as K

            """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.

            Applies the sigmoid activation function. For small values (<-5),
            `sigmoid` returns a value close to zero, and for large values (>5)
            the result of the function gets close to 1.

            Sigmoid is equivalent to a 2-element Softmax, where the second element is
            assumed to be zero. The sigmoid function always returns a value between
            0 and 1.

            For example:

            >>> a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
            >>> b = tf.keras.activations.sigmoid(a)
            >>> b.numpy()
            array([2.0611537e-09, 2.6894143e-01, 5.0000000e-01, 7.3105860e-01,
                     1.0000000e+00], dtype=float32)

            Args:
                x: Input tensor.

            Returns:
                Tensor with the sigmoid activation: `1 / (1 + exp(-x))`.
            """
            fx = K.exp(-x) / K.sum(K.exp(-x))
            output = (K.exp(x) / (1 + K.exp(-x) ** 2)) + fx
            # output = nn.sigmoid(x)
            # # Cache the logits to use for crossentropy loss.
            # output._keras_logits = x  # pylint: disable=protected-access
            return output

        x = layers.Conv2D(num_classes, (1, 1), activation=sigmoid, padding='same')(x)

        return x

    vv = np.load('saved data/gt.npy')

    def mask_rcnn(input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)

        # Backbone network (you can use any pre-trained backbone like ResNet, etc.)

        backbone = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        backbone = layers.Dropout(0.5)(backbone)
        backbone = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.MaxPooling2D(pool_size=(2, 2))(backbone)

        # Add more layers based on the backbone architecture
        backbone = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.Dropout(0.5)(backbone)
        backbone = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.MaxPooling2D(pool_size=(2, 2))(backbone)

        # Add more layers based on the backbone architecture
        backbone = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.Dropout(0.5)(backbone)
        backbone = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.MaxPooling2D(pool_size=(2, 2))(backbone)

        # Add more layers based on the backbone architecture
        backbone = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.Dropout(0.5)(backbone)
        backbone = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(backbone)
        backbone = layers.MaxPooling2D(pool_size=(2, 2))(backbone)  #########  # encoding

        # Region Proposal Network (RPN)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(backbone)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(rpn)
        rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(rpn)

        # Region Proposal Network Output
        rpn_class = layers.Conv2D(2, (1, 1), activation='softmax')(rpn)
        rpn_bbox = layers.Conv2D(4, (1, 1))(rpn)

        # Mask Prediction Head
        mask_pred = mask_head(rpn, num_classes)

        model = models.Model(inputs=inputs, outputs=[rpn_class, rpn_bbox, mask_pred])
        return model

    # Usage
    num_classes = 1  # Number of classes, e.g., tumor
    input_shape = (im.shape[0], im.shape[1], 1)  # Input shape of your grayscale images
    model = mask_rcnn(input_shape, num_classes)

    # print(model.summary())
    from tensorflow.keras.utils import plot_model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'rpn_class': 'binary_crossentropy',
                        'rpn_bbox': 'smooth_l1',
                        'mask_pred': 'binary_crossentropy'},

                  )
    from tensorflow.keras.models import load_model
    model.load_weights('cm_Rcnn/trained_mask_rcnn_model.h5')

    test_image = im
    test_image = cv2.resize(test_image, (im.shape[0], im.shape[1]))
    test_image = np.reshape(test_image, [1, im.shape[0], im.shape[1]])
    # # test_image = cv2.imread("path/to/image.jpg")
    # test_image = test_image / 255.0  # Normalize to [0, 1]
    # test_image = np.expand_dims(test_image, axis=0)

    # Perform segmentation
    rpn_class, rpn_bbox, mask_pred = model.predict(test_image)
    # mask_pred = (mask_pred * 255).astype('uint8')
    mask_pred = cv2.resize(mask_pred[0, :, :, 0], (im.shape[0], im.shape[1]))
    pred_im = vv * (mask_pred * 255).astype('uint8')
    # cv2.imshow('pred_im', pred_im)
    pred_im = pred_im * im
    # threshold=0.5
    # binary_mask = vv*((mask_pred > threshold).astype(np.uint8))

    # out=(mask_pred*255).astype('uint8')
    # threshold=np.mean(mask_pred)
    # binary_mask = (mask_pred > threshold).astype(np.uint8)
    # cv2.imshow('im', test_image[0, :, :].astype('uint8'))
    # cv2.imshow('binary_mask', binary_mask.astype('uint8'))
    return pred_im


def prop_aHE(image):

    # step1:Invert the image
    # Invert the image
    # inverted_image = 255 - image
    # cv2.imshow('inverted_image Image', inverted_image)
    # inverted_image=image
    # step2: dehazing
    def dark_channel(image, window_size):
        # Compute the dark channel of the image using a minimum filter
        min_channel = cv2.erode(image, np.ones((window_size, window_size)), iterations=1)
        return np.min(min_channel)

    def estimate_atmospheric_light(dark_channel, top_percentage):
        # Flatten the dark channel and sort in descending order
        flat_dark_channel = dark_channel.flatten()
        flat_dark_channel.sort()
        num_pixels = len(flat_dark_channel)

        # Estimate the number of pixels to consider based on the top_percentage
        num_pixels_to_consider = int(num_pixels * top_percentage / 100.0)

        # Take the maximum intensity among the top_percentage pixels
        atmospheric_light = np.max(flat_dark_channel[-num_pixels_to_consider:])
        return atmospheric_light

    def dehaze(image, window_size=15, top_percentage=0.001, t_min=0.1):
        # Convert the image to float32 for calculations
        image = image.astype(np.float32) / 255.0

        # Calculate the dark channel of the image
        dark_ch = dark_channel(image, window_size)

        # Estimate the atmospheric light in the scene
        A = estimate_atmospheric_light(dark_ch, top_percentage)

        # Calculate the transmission map
        transmission = 1.0 - top_percentage * dark_ch / A

        # Clip the transmission map to ensure values are within [t_min, 1]
        transmission = np.clip(transmission, t_min, 1)

        # Calculate the scene radiance (dehazed image)
        dehazed_image = (image - A) / transmission + A

        # Clip the dehazed image pixel values to ensure they are in the valid range
        dehazed_image = np.clip(dehazed_image, 0, 1)

        return (dehazed_image * 255).astype(np.uint8)

    # Load the hazy image
    hazy_image = image#cv2.imread('path_to_hazy_image.jpg')

    # Dehaze the image
    dehazed_image = dehaze(hazy_image)

    # Display the hazy and dehazed images
    # cv2.imshow('Dehazed Image', dehazed_image)

    i_dehazed_image =dehazed_image
    def gray_world_assumption(image):
        # Calculate the average color values for each channel
        avg_color_per_channel = np.mean(image, axis=(0, 1))

        # Calculate the overall average color value
        avg_color = np.mean(avg_color_per_channel)

        # Calculate the scaling factors for each channel
        scale_factors = avg_color / avg_color_per_channel

        # Scale the color channels using the scaling factors
        corrected_image = np.multiply(image, scale_factors).astype(np.uint8)

        return corrected_image

    corrected_image = gray_world_assumption(i_dehazed_image)
    # cv2.imshow('corrected_image', corrected_image)

    # Define the scaling factor (alpha) and shift value (beta)
    alpha = 1.2
    beta = 30
    def linear_transform(image, alpha, beta):
        # Perform the linear transformation
        transformed_image = (image.astype(np.float32) * alpha) + beta

        # Clip the pixel values to ensure they are in the valid range [0, 255]
        transformed_image = np.clip(transformed_image, 0, 255)

        # Convert the image back to the unsigned 8-bit integer type
        transformed_image = transformed_image.astype(np.uint8)

        return transformed_image
    last =linear_transform(corrected_image,alpha,beta)
    # cv2.imshow('last', last)
    return last


def c_adaptive_histogram(image):

    # Calculate the histogram using OpenCV function
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate the cumulative sum of the histogram
    cdf = hist.cumsum()

    # Normalize the CDF to bring the values between 0 and 1
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Create an intensity mapping using the normalized CDF
    mapping = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    # Reshape the mapping to match the original image shape
    equalized_image = mapping.reshape(image.shape)
    # Convert the equalized image to 8-bit for visualization
    equalized_image = np.uint8(equalized_image)
    # # Display the images using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(equalized_image, cmap='gray')
    # plt.title('Equalized Image')
    return equalized_image


def median_filter(img):
    # resize #
    sign = img.astype('uint8')
    # sign = cv2.resize(img, (64, 64))  # sign = cv2.resize(img, (256, 256))
    filtered_im = cv2.medianBlur(sign, 5)
    return filtered_im

class FCM:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []

        # u = np.random.dirichlet(np.ones(C), size=N)
        r = np.random.RandomState(self.random_state)
        u = r.rand(N,C)

        u = u / np.tile(u.sum(axis=1)[np.newaxis].T,C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            from scipy.linalg import norm
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        from scipy.spatial.distance import cdist
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        # alpha=reyi_entropy(temp)
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_

        return 1 / denominator_.sum(2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X, self.centers)
        return np.argmax(u, axis=-1)

def conv_dfc(pp_im):
        F1 = FCM(n_clusters=4)
        X = pp_im.flatten()
        X = X.reshape(-1, 1)
        F1.fit(X)
        fcm_centers = F1.centers
        fcm_labels = F1.u.argmax(axis=1)
        fcm_ot = fcm_labels.reshape(pp_im.shape[0], pp_im.shape[1])
        return fcm_ot

def mean_filter(image, kernel_size):
    # Define a kernel for the mean filter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # Apply the mean filter to the image
    result = cv2.filter2D(image, -1, kernel)

    return result


NAM = ['ORG-IMG', 'conventional_median', 'preprocessed_proposed_median', 'mean_filter','gaussian_filter','winer_filter']

seg_NAM=['FCM','unet','conventional_mRCNN','proposed']

I1,I2,I3,I4,I5,I6 =[],[],[],[],[],[]

I11,I12,I13,I14,I15,I16 =[],[],[],[],[],[]

fin_lab, ii = [], 0
idxx = 1
DB_path = ['Database/']
ch = 0
casepath = os.path.join(os.path.curdir, DB_path[0])
cases_all = [f for f in os.listdir(casepath) if ~os.path.isfile(os.path.join(casepath, f))]
ii = 0
# cv_size = 512
vv = np.load('saved data/fin_label.npy')
fin_feat, fin_feat_c = np.zeros([len(vv), 300]), np.zeros([len(vv), 300])
fin_label = []
for n_case in range(len(cases_all)):
    filefullpath = os.path.join(os.path.curdir, DB_path[ch], cases_all[n_case])
    print(filefullpath)

    cases_all_p = [f for f in os.listdir(filefullpath) if ~os.path.isfile(os.path.join(filefullpath, f))]
    fin_label.extend(np.zeros(len(cases_all_p)) + n_case)
    for n_case_p in range(len(cases_all_p)):

        path = os.path.join(os.path.curdir, DB_path[ch], cases_all[n_case], cases_all_p[n_case_p])
        print(path)
        im = cv2.imread(path)
# path = 'Database'
# gt_path = os.listdir(path)[0]
# gt_li = os.listdir(os.path.join(path, gt_path))
# gt_li.sort(key=lambda f: int(re.sub('\D', '', f)))
#
# for type in range(len(DB_path)):
#     casepath = os.path.join(os.path.curdir, DB_path[type])
#     cases_all = [f for f in os.listdir(casepath) if ~os.path.isfile(os.path.join(casepath, f))]
#     cases_all.sort(key=lambda f: int(re.sub('\D', '', f)))
#     for n_case in range(1, len(cases_all),1):
        if idxx>5: break
        else:
            print(n_case)
            fin_lab.append(type)
            # gt_ = os.path.join(os.path.curdir, path, gt_path, gt_li[n_case])
            # mk =nib.load(gt_)
            # mk_data =mk.get_fdata()
            # print(gt_, mk_data.shape)
            #
            # index=int(mk_data.shape[2]/2)
            #
            # # for kk in range((mk_data.shape[2])):
            # #     mk_ =mk_data[:,:,kk]
            # mk_ = mk_data[:, :, index]
            # mk_= cv2.resize(mk_,(256,256))
            # if np.sum(mk_)==0:
            #     index = int(mk_data.shape[2] / 2) +30
            #     mk_ = mk_data[index]
            #     mk_ = cv2.resize(mk_, (256, 256))
            # np.save('saved data/gt.npy',mk_)
            # plt.figure(100)
            # plt.imshow(mk_)
            # plt.axis('off')
                # plt.savefig('processed/'+str(kk)+'gt.png')

            # filefullpath = os.path.join(casepath, cases_all[n_case])  # read full directory of database1 & database2
            # # read image #
            # nii_pth = nib.load(filefullpath)
            # nii_data = nii_pth.get_fdata()
            # print(filefullpath,nii_data.shape)
            #
            # image =nii_data[:,:,index]
            # # for kk in range((mk_data.shape[2])):
            # #     image = nii_data[:, :, kk]
            image= im# image = cv2.resize(im,(cv_size,cv_size))
            cv_size = image.shape[1]
            img_d = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            plt.figure(0)
            plt.imshow(image)
            plt.axis('off')
            cv2.imwrite('image_results/' + NAM[0] + str(idxx) + '.png',image)
            # plt.savefig('image_results/' + NAM[0] + str(idxx) + '.png',format="tif")

            # plt.savefig('processed/' + str(kk) + 'input.png')

            # plt.savefig('image_results/segment/' + NAM[0] + str(idxx) + '.tif',format="tif")
            # plt.figure(1)
            # plt.imshow(image)
            # plt.axis('off')
            #

            I1.append(image)
            # plt.imshow(image)
            #########+++++++++++ STEP1: PreProcessing +++++++++++#####

            # --------<<<<<<<<<<STEP1:Pre-Processing  >>>>>>>>>>>>-------------#
            imm_c = cv2.medianBlur(image, 5)
            plt.figure(1)
            plt.imshow(imm_c)
            plt.axis('off')
            cv2.imwrite('image_results/' + NAM[1] + str(idxx) + '.png',imm_c)

            # plt.savefig('image_results/' + NAM[1] + str(idxx) + '.png')


            # improved median filter #
            def Imedian_filter(data):
                from skimage import color
                if len(data.shape) == 3:
                    data = color.rgb2gray(data)
                filter_size = 3
                # center element f(i,j)
                i, j = int(data.shape[0] / 2), int(data.shape[0] / 2)
                center = data[i][j]
                from scipy.stats import gmean
                temp = []
                indexer = filter_size // 2
                data_final = []
                data_final = np.zeros((len(data), len(data[0])))
                ###  filtering mask improvement ######
                MP1 = np.median(data) - np.min(data)  ### updated mask pixel
                MP2 = np.median(data) - np.max(data)

                # filter_size=n
                for i in range(len(data)):

                    for j in range(len(data[0])):
                        for z in range(filter_size):  # window size
                            if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                                for c in range(filter_size):
                                    temp.append(0)
                            else:
                                if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                                    temp.append(0)
                                else:
                                    for k in range(filter_size):
                                        temp.append(data[i + z - indexer][j + k - indexer])
                        # weighted gemoetric mean#
                        # wt =np.random.dirichlet(np.ones(len(temp)), size=1).flatten().tolist()
                        # sum_squared_weights = sum(w ** 2 for w in wt)
                        # sum_weighted_squares = sum(w * (v ** 2) for v, w in zip(temp, wt))
                        #
                        # WQM = math.sqrt(sum_weighted_squares / sum_squared_weights)
                        # Thresh = 0.5* WQM
                        NVF = (np.var(data) / (filter_size + np.pi / 2 - 1)) * (np.pi / 2)  # noise variance of filter
                        # wPm= weighted_power_mean(temp,wt,2)  ######## power
                        BP1 = data[i][j] - np.min(data)  ### updated
                        BP2 = data[i][j] - np.max(data)
                        # compare each pixel with wgn
                        if (MP1 > 0 and MP2 < 0):
                            data[i][j] = np.median(temp)
                        elif (BP1 > 0 and BP2 < 0):
                            data[i][j] = data[i][j]
                        elif data[i][j] < NVF:  ############### updated
                            data[i][j] = np.median(temp)  # median value
                        else:
                            data[i][j] = data[i][j]  # orginal value
                        temp.sort()
                        data_final[i][j] = temp[len(temp) // 2]
                        temp = []
                data_final = (data_final * 255).astype('uint8')
                return data_final


            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im[:, :, 0])
            vv = cv_size-20
            x0 = int(maxLoc[0]) - vv
            y0 = int(maxLoc[1]) - vv
            x1 = int(maxLoc[0]) + vv
            y1 = int(maxLoc[1]) + vv
            # roi_im = image[y0+80:y1, x0:x1]
            roi_im = image[280:im.shape[0] - 80, :] # original size
            rr=cv2.cvtColor(roi_im,cv2.COLOR_BGR2GRAY)
            ret4, th = cv2.threshold(rr, 80, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            p_med  = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            # p_med = cv2.medianBlur(th, 3)

            # p_med = Imedian_filter(roi_im)
            # plt.figure(3)
            # plt.imshow(p_med)
            input =image

            # adptive  histogram equalization #
            # filtered_im = c_adaptive_histogram(input)

            # improved adptive  histogram equalization #
            # filtered_im_p = prop_aHE(input)

            I2.append(imm_c)
            I3.append(p_med)
            # plt.figure(2)
            # plt.imshow(imm_c)
            # plt.axis('off')

            # plt.savefig('image_results/DB'+str(db)+'/' + NAM[1] + str(idxx) + '.tif',format="tif")
            plt.figure(3)
            plt.imshow(p_med, cmap='gray')
            plt.axis('off')
            # plt.savefig('image_results/' + NAM[2] + str(idxx) + '.png')
            cv2.imwrite('image_results/' + NAM[2] + str(idxx) + '.png',p_med)

            # plt.savefig('image_results/DB'+str(db)+'/' + NAM[2] + str(idxx) + '.tif',format="tif")

            # median #

            mean_im = mean_filter(image,kernel_size=7)
            I6.append(mean_im)
            plt.figure(4)
            plt.imshow(mean_im)
            plt.axis('off')
            cv2.imwrite('image_results/' + NAM[3] + str(idxx) + '.png',mean_im)

            # plt.savefig('image_results/' + NAM[3] + str(idxx) + '.png')


            # Gaussian Blur
            img=image
            Gaussian = cv2.GaussianBlur(img.astype('uint8'), (9, 9), 0)
            # cv2.imshow('Gaussian Blurring', Gaussian)
            I4.append(Gaussian)
            plt.figure(5)
            plt.imshow(Gaussian)
            plt.axis('off')
            # plt.savefig('image_results/' + NAM[4] + str(idxx) + '.png')
            cv2.imwrite('image_results/' + NAM[4] + str(idxx) + '.png',Gaussian)



            # plt.savefig('image_results/DB'+str(db)+'/' + NAM[3] + str(idxx) + '.tif',format="tif")

            from scipy.signal import wiener
            # g_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale image
            wiener_im = wiener(input[:,:,0], (9, 9)).astype('uint8')
            plt.figure(6)
            plt.imshow(wiener_im, cmap='gray')
            plt.axis('off')
            # plt.savefig('image_results/' + NAM[5] + str(idxx) + '.png')
            cv2.imwrite('image_results/' + NAM[5] + str(idxx) + '.png',wiener_im)

            # plt.savefig('image_results/DB'+str(db)+'/' + NAM[4] + str(idxx) + '.tif',format="tif")
            I5.append(wiener_im)



            # fcm_im = conv_dfc(input)
            # I11.append(fcm_im)
            # plt.figure(101)
            # plt.imshow(fcm_im, cmap='gray')
            # plt.imshow(fcm_im)
            # plt.axis('off')
            # # plt.savefig('image_results/segment/' + seg_NAM[0] + str(idxx) + '.tif',format="tif")
            #
            #
            # seg_unet = cunet(input)
            # I12.append(seg_unet)
            # plt.figure(102)
            # plt.imshow(seg_unet)
            # plt.axis('off')
            # # plt.savefig('image_results/segment/' + seg_NAM[1] + str(idxx) + '.tif',format="tif")
            #
            # # conventional RCNN#
            # srcnn =conv_RCNN(filtered_im_p,input).astype('uint8')
            # I13.append(srcnn)
            # plt.figure(103)
            # plt.imshow(srcnn,cmap='gray')
            # plt.axis('off')
            # # plt.savefig('image_results/segment/' + seg_NAM[2] + str(idxx) + '.tif',format="tif")
            #
            # s_im = prop_cm_RCNN(filtered_im_p,input).astype('uint8')
            # I14.append(s_im)
            # plt.figure(104)
            # plt.imshow(s_im)
            # plt.axis('off')
            # plt.savefig('image_results/segment/' + seg_NAM[-1] + str(idxx) + '.tif',format="tif")
            idxx+=1
            plt.pause(2)
            plt.close()


np.save('saved data/I11',np.array(I11))
np.save('saved data/I12',np.array(I12))
np.save('saved data/I13',np.array(I13))
np.save('saved data/I14',np.array(I14))

np.save('saved data/I1',np.array(I1))
np.save('saved data/I2',np.array(I2))
np.save('saved data/I3',np.array(I3))
np.save('saved data/I4',np.array(I4))
np.save('saved data/I5',np.array(I5))
np.save('saved data/I6',np.array(I6))

def mean_filter(image, kernel_size):
    # Define a kernel for the mean filter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # Apply the mean filter to the image
    result = cv2.filter2D(image, -1, kernel)
    return result

db=1
sspr=np.load('saved data/ssim_psnr'+'.npy')
df = pd.DataFrame(sspr, columns=['PSNR', 'SSIM'], index=['Median', 'Gaussian', 'Weiner', 'conventional', 'proposed'])
print(df)
if rn==1:
    img = np.load('/pre_evaluated/DATA.npy', allow_pickle=True)
    pre_imgc = np.load('/pre_evaluated/pre_data_c.npy', allow_pickle=True)
    pre_imgp = np.load('/pre_evaluated/pre_data_p.npy', allow_pickle=True)
    segimg = np.load('/pre_evaluated/images.npy', allow_pickle=True)
    im=img.copy()

    IMG = [img, pre_imgc, pre_imgp, img] #segimg[1]]
    NAM = ['ORG-IMG', 'pre_processed_con_img', 'pre_processed_prop_img', 'seg_img']


    I1,I2,I3,I4,I5 =[],[],[],[],[]
    for idx, (imgg, name) in enumerate(zip(IMG, NAM)):
        for idxx, ii in enumerate([20, 700, 690, 25, 130]):
            img = imgg[ii]
            if idx==0:
                I1.append(img)

            if idx == 1:
                I4.append(img)

            if idx == 2:
                I5.append(img)

            if idx == 3:  # seg
                z1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                simg = cv2.threshold(z1, 0, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                simg[simg == 0] = 1
                simg[simg > 1] = 0
                plt.imshow(simg, cmap='gray')
                plt.axis('off')
                # plt.savefig('../Results_JC/' + NAM[idx] + str(idxx) + '.jpg')

                # Gaussian Blur
                Gaussian = cv2.GaussianBlur(img, (11, 11), 0)
                # cv2.imshow('Gaussian Blurring', Gaussian)
                I2.append(Gaussian)
                plt.imsave('../Results_JC/' + 'Gaussian filter' + str(idxx) + '.jpg', Gaussian)

                from scipy.signal import wiener

                g_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale image
                wiener_im = wiener(g_im, (35, 35)).astype('uint8')
                # cv2.imshow('winer filter', wiener_im)
                I3.append(wiener_im)
                plt.imsave('../Results_JC/' + 'wiener filter' + str(idxx) + '.jpg', wiener_im)

            else:
                img = cv2.resize(img, (256, 256))
                plt.imsave('../Results_JC/' + NAM[idx] + str(idxx) + '.jpg', img)

h = 5
# np.save('../saved data/I_list',np.array([I1,I2,I3,I4,I5]))




import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io


def calculate_psnr(original, compressed):
    # # Read images
    # original = io.imread(original_img)
    # compressed = io.imread(compressed_img)

    # Calculate PSNR
    psnr_value = psnr(original, compressed)

    return psnr_value


def calculate_ssim(original, compressed):
    # Read images
    # original = io.imread(original_img)
    # compressed = io.imread(compressed_img)

    # Calculate SSIM
    ssim_value = ssim(original, compressed, multichannel=True)

    return ssim_value


def ps_ssim_values(original_image, compressed_image):
    if original_image.shape != compressed_image.shape :
        from skimage import color, restoration
        original_image = color.rgb2gray(original_image)

    psnr_result = calculate_psnr(original_image, compressed_image)
    ssim_result = calculate_ssim(original_image, compressed_image)
    return psnr_result, ssim_result

an=0
if an==1:
    data = np.load('saved data/I1.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)
    data1 = np.load('saved data/I2.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)
    data2 = np.load('saved data/I3.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)
    data3 = np.load('saved data/I4.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)
    data4 = np.load('saved data/I5.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)
    data5 = np.load('saved data/I6.npy', allow_pickle=True)#np.load('pre_evaluated/data.npy', allow_pickle=True)

    pre_img = [data1,data2,data3,data4,data5]#np.load('pre_evaluated/pre_data.npy', allow_pickle=True)
    dd = np.zeros([len(pre_img), 2])
    # for idx, (imgg, name) in enumerate(zip(data, pre_img)):
    #     for idxx, ii in enumerate([0, 48, 70, 95, 500]):
    # for i in range(len(data)):
    for idx, (imgg, name) in enumerate(zip(data, pre_img)):
            print(imgg.shape, name[idx].shape)
            org = imgg.astype('uint8')
            # org = imgg[ii]
            pre = name[idx]
            dd[idx, :] = ps_ssim_values(org, pre)

    sspr =np.load('saved data/ssim_psnr.npy')
    import pandas as pd
    df =pd.DataFrame(sspr,columns=['PSNR','SSIM'],index=['Median','Gaussian', 'Weiner','conventional','proposed'])


    # df.to_csv('../Results_/PSNRSSIM.csv')

    def jaccard_dice_acc():
        def jagard_coeff(a, b, c, d, e):
            methods = [b, c, d, e]
            gt = a
            methd, methd_ac = [], []
            for jj, method in enumerate(methods):
                img, im_ac = [], []
                for ii, i in enumerate(method):
                    true = gt.copy()
                    true = true[ii]
                    pred = i
                    score = jaccard_score(true.flatten(), pred.flatten(), average="micro")
                    ac_score = accuracy_score(true.flatten(), pred.flatten())
                    img.append(score)
                    im_ac.append(ac_score)
                methd.append(img)
                methd_ac.append(im_ac)
            if rn == 1:
                np.save('saved data/jaccard', methd)
                np.save('saved data/seg_acc', methd_ac)

        def dice_coeff(a, b, c, d, e):
            methods = [b, c, d, e]
            gt = a
            methd = []
            for jj, method in enumerate(methods):
                img = []
                for ii, i in enumerate(method):
                    true = gt.copy()
                    true = true[ii]
                    pred = i
                    k = 255
                    intersection = np.sum(pred[true == k]) * 2.0
                    dice = intersection / (np.sum(pred) + np.sum(true))
                    img.append(dice)
                methd.append(img)
            if rn == 1: np.save('saved data/dice', methd)

        if rn == 1:
            i1, i2, i3, i4, i5 = np.load('saved data/i1.npy'), np.load('saved data/i2.npy'), np.load(
                'saved data/i3.npy'), np.load('saved data/i4.npy'), np.load('saved data/i5.npy')

            dice_coeff(i1, i2, i3, i4, i5)
            jagard_coeff(i1, i2, i3, i4, i5)

        dja = np.load('saved data/seg_ot_dja' + '.npy')
        mtd =  seg_NAM #['fcm', 'kmeans', 'conventional DJ', 'proposed DJ']
        name2 = ['dice', 'jaccard', 'segmentation accuracy']
        df2 = pd.DataFrame(dja, columns=mtd, index=name2)
        print(df2)


    jaccard_dice_acc(ii)

#
# for type in range(len(DB_path)):
#     casepath = os.path.join(os.path.curdir, DB_path[type])
#     cases_all = [f for f in os.listdir(casepath) if ~os.path.isfile(os.path.join(casepath, f))]
#     cases_all.sort(key=lambda f: int(re.sub('\D', '', f)))
#
#     for n_case in range(1, len(cases_all),1):
#         print('n_case>>>>>',n_case)
#         if idxx>5: break
#         else:
#             gt_ = os.path.join(os.path.curdir, path, gt_path, gt_li[n_case])
#             print(gt_)
#             mk =nib.load(gt_)
#             mk_data =mk.get_fdata()
#             index=int(mk_data.shape[2]/2)
#             mk_ =mk_data[index]
#             mk_= cv2.resize(mk_,(256,256))
#             if np.sum(mk_)==0:
#                 index = int(mk_data.shape[2] / 2) +20
#                 mk_ = mk_data[index]
#                 mk_ = cv2.resize(mk_, (256, 256))
#             # np.save('saved data/gt.npy',mk_)
#             # plt.figure(100)
#             # plt.imshow(mk_)
#             # plt.axis('off')
#
#
#             filefullpath = os.path.join(casepath, cases_all[n_case])  # read full directory of database1 & database2
#             print(filefullpath)