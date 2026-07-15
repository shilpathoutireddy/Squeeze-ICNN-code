
##########  import all packages ###########
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').disabled = True
import warnings, glob
warnings.filterwarnings("ignore")
from warnings import simplefilter
import cv2, matplotlib.pyplot as plt, numpy as np,tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate,LeakyReLU,Conv2D,GlobalMaxPool2D, MaxPool2D, GlobalAveragePooling2D, Dropout,Flatten,Dense
from tensorflow.keras.layers import Layer,Add
rn=0
from tensorflow.keras.applications import VGG16,ResNet50,InceptionV3
VGG_model = VGG16(weights='imagenet', include_top=False)
Resnet_model = ResNet50(weights='imagenet', include_top=False)
Inceptionv3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
from  other.Confusion_matrix import *
from  other.root import *
from tensorflow.keras import layers
from pytictoc import TicToc
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from other import popup

################# Function #################
def bp1_(X_train, Y_train, X_test, Y_test, db):
    ln = len(set(Y_train.flatten()))
    Lab = np.concatenate((Y_train, Y_test))
    cnn_X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
    cnn_X_test = X_test.reshape((X_test.shape[0], 1, 1, X_test.shape[1]))
    y_train = to_categorical(Y_train)
    model = InceptionV3(weights='imagenet')
    model = Sequential()
    model.add(Conv2D(8, (1, 1), padding='valid', input_shape=cnn_X_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(ln, activation='softmax'))
    model.add(Dropout(0.2))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    sv_weight = np.array(model.get_weights())  # save weights in a np.array of np.arrays
    model.fit(cnn_X_train, y_train, epochs=ep, batch_size=28, verbose=0)
    print(model)
    pred = model.predict(cnn_X_test)
    y_pred = array(np.argmax(pred, axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
    return out, y_pred, cm[0], cm[1]

one = (1, 1)
two = (2, 2)
three = (3, 3)
five = (5, 5)
seven = (7, 7)
thirteen = (13, 13)
compression= 0.1

def bp2_(X_train, Y_train, X_test, Y_test,db):
    ln = len(set(Y_train.flatten()))
    Lab = np.concatenate((Y_train, Y_test))
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    weight = model.get_weights()
    train_lab = to_categorical(Y_train)
    model.fit(X_train, train_lab, epochs=ep, batch_size=64, verbose=0)
    y_pred = model.predict(X_test)
    y_pred = array(np.argmax(y_pred, axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
    return out, y_pred, cm[0], cm[1]


def cnn_net(X_train, Y_train, X_test, Y_test, db):
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    ln = len(set(Y_train.flatten()))
    Lab = np.concatenate((Y_train, Y_test))
    cnn_X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
    cnn_X_test = X_test.reshape((X_test.shape[0], 1, 1, X_test.shape[1]))
    y_train = to_categorical(Y_train)
    model = InceptionV3(weights='imagenet')
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding='valid', input_shape=cnn_X_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(ln, activation='softmax'))
    model.add(Dropout(0.2))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    sv_weight = np.array(model.get_weights())  # save weights in a np.array of np.arrays
    model.fit(cnn_X_train, y_train, epochs=ep, batch_size=28, verbose=0)
    print(model)
    pred = model.predict(cnn_X_test)
    y_pred = array(np.argmax(pred, axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
    return out, y_pred, cm[0], cm[1]

def dense_net(X_train , y_train,X_test, y_test,db):
        t = TicToc()
        t.tic()
        Input ,compression= 1,0.1
        n = [0, 1]
        X_train = np.resize(X_train, (X_train.shape[0], 32, 32, 3))
        X_test = np.resize(X_test, (X_test.shape[0], 32, 32, 3))
        image_height, image_width, channel_size = X_train.shape[1], X_train.shape[2], X_train.shape[3]
        num_classes = len(np.unique(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        def denseblock(input, num_filter=12, dropout_rate=0.2):
            global compression
            temp = input
            for _ in range(l):
                BatchNorm = layers.BatchNormalization()(temp)
                relu = layers.Activation('relu')(BatchNorm)

                Conv2D_3_3 = layers.Conv2D(int(num_filter * compression), (3, 3), use_bias=False, padding='same')(relu)
                if dropout_rate > 0:
                    Conv2D_3_3 = layers.Dropout(dropout_rate)(Conv2D_3_3)
                concat = layers.Concatenate(axis=-1)([temp, Conv2D_3_3])
                temp = concat
            return temp

        ## transition Blosck
        def transition(input, num_filter=12, dropout_rate=0.2):
            global compression
            BatchNorm = layers.BatchNormalization()(input)
            relu = layers.Activation('relu')(BatchNorm)
            Conv2D_BottleNeck = layers.Conv2D(int(num_filter * compression), (1, 1), use_bias=False, padding='same')(
                relu)
            if dropout_rate > 0:
                Conv2D_BottleNeck = layers.Dropout(dropout_rate)(Conv2D_BottleNeck)
            avg = layers.AveragePooling2D(pool_size=(2, 2))(Conv2D_BottleNeck)
            return avg

        def concatenate(inputs, axis=-1, shape=None, **kwargs):
            """Functional interface to the `Concatenate` layer.

          >>> x = np.arange(20).reshape(2, 2, 5)
          >>> print(x)
          [[[ 0  1  2  3  4]
            [ 5  6  7  8  9]]
           [[10 11 12 13 14]
            [15 16 17 18 19]]]
          >>> y = np.arange(20, 30).reshape(2, 1, 5)
          >>> print(y)
          [[[20 21 22 23 24]]
           [[25 26 27 28 29]]]
          >>> tf.keras.layers.concatenate([x, y],
          ...                             axis=1)
          <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
          array([[[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [20, 21, 22, 23, 24]],
               [[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [25, 26, 27, 28, 29]]])>

          Args:
              inputs: A list of input tensors (at least 2).
              axis: Concatenation axis.
              **kwargs: Standard layer keyword arguments.

          Returns:
              A tensor, the concatenation of the inputs alongside axis `axis`.
          """
            return Concatenate(axis=axis, **kwargs)(inputs), shape

        # output layer
        def output_layer(input):
            # updation
            global compression
            BatchNorm = layers.BatchNormalization()(input)
            relu = layers.Activation('relu')(BatchNorm)
            global_average_pooling = GlobalAveragePooling2D()(relu)  # global average pooling
            global_max_pooling = GlobalMaxPool2D()(relu)  # global max pooling
            concat_lay = concatenate([global_average_pooling, global_max_pooling], shape=relu)[Input]
            AvgPooling = layers.AveragePooling2D(pool_size=(2, 2))(concat_lay)
            flat = layers.Flatten()(AvgPooling)
            output = layers.Dense(num_classes, activation='softmax')(flat)
            return output

        l = 7
        input = layers.Input(shape=(image_height, image_width, channel_size,))
        First_Conv2D = layers.Conv2D(30, (3, 3), use_bias=False, padding='same')(input)

        First_Block = denseblock(First_Conv2D, 30, 0.5)
        First_Transition = transition(First_Block, 30, 0.5)
        Last_Block = denseblock(First_Transition, 30, 0.5)
        output = output_layer(Last_Block)

        model = Model(inputs=[input], outputs=[output])

        # determine Loss function and Optimizer
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(height_shift_range=0.1, width_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                     horizontal_flip=True)
        datagen.fit(X_train)
        data = datagen.flow(x=X_train, y=y_train, batch_size=50)
        step_size = X_train.shape[0] // 10
        y_predict = array(np.argmax(model.predict(X_test), axis=-1), axis=[0, 1])
        out, cm = multi_confu_matrix(y_test, y_predict, ll, mm)
        return out, y_predict, cm[0], cm[1]


def bigru_(X_train, Y_train, X_test, Y_test, db):
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words, n = 5000, 2
    train_X = X_train.reshape(-1, X_train.shape[1], 1)
    test_X = X_test.reshape(-1, X_train.shape[1], 1)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / train_X.max()
    test_X = test_X / train_X.max()
    # Change the labels from categorical to one-hot encoding
    train_Y = to_categorical(Y_train)
    test_Y = to_categorical(Y_test)
    Lab = np.concatenate((Y_train, Y_test))
    # Final evaluation of the model
    model = Sequential()
    model.add(Bidirectional(GRU(124, return_sequences=True), input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.5))
    # number of features on the output
    model.add(Dropout(0.2))
    model.add(Dense(train_Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    rnn_train = model.fit(train_X, train_Y, batch_size=128, epochs=ep, verbose=0)
    sv_weight = np.array(model.get_weights())  # save weights in a np.array of np.arrays
    pred = model.predict(test_X)
    y_pred = array(np.argmax(pred, axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, ll, mm)
    return out, y_pred, cm[0], cm[1]


def lstm_(X_train, Y_train, X_test, Y_test, db):
    ln = len(set(Y_train.flatten()))
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    train_X = X_train.reshape(-1, X_train.shape[1], 1)
    test_X = X_test.reshape(-1, X_train.shape[1], 1)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / train_X.max()
    test_X = test_X / train_X.max()
    Lab = np.concatenate((Y_train, Y_test))
    # Change the labels from categorical to one-hot encoding
    train_Y = to_categorical(Y_train)
    test_Y = to_categorical(Y_test)
    embedding_vecor_length = 32

    # Final evaluation of the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    # number of features on the output
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    rnn_train = model.fit(train_X, train_Y, batch_size=128, epochs=ep, verbose=0)
    pred = model.predict(test_X)
    y_pred = array(np.argmax(pred, axis=1), axis=[2, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
    return out, y_pred, cm[0], cm[1]


def dcnn_(X_train, Y_train, X_test, Y_test, db):
    ln = len(set(Y_train.flatten()))
    Lab = np.concatenate((Y_train, Y_test))
    yy = Y_train
    cnn_X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
    cnn_X_test = X_test.reshape((X_test.shape[0], 1, 1, X_train.shape[1]))
    model = Sequential()
    model.add(Conv2D(32, (1, 1), padding='valid', input_shape=cnn_X_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\n DCNN model')
    model.summary()
    model.fit(cnn_X_train, Y_train, epochs=ep, batch_size=28, verbose=0)
    print(model)
    pred = model.predict(cnn_X_test)
    y_pred = array(np.argmax(pred, axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
    return out, y_pred, cm[0], cm[1]


def load_data():
    xx =np.load('saved data/fin_feat.npy')
    yy = np.load('saved data/fin_feat_c.npy')
    zz = np.load('saved data/fin_label.npy')
    zz[zz>1]=1
    return xx,zz


def full_analysis():

    def read_feat_extract():
        DB_path = ['Database/']
        ch=0
        casepath = os.path.join(os.path.curdir, DB_path[0])
        cases_all = [f for f in os.listdir(casepath) if ~os.path.isfile(os.path.join(casepath, f))]
        ii = 0
        cv_size=128
        vv = np.load('saved data/fin_label.npy')
        fin_feat,fin_feat_c=np.zeros([len(vv),300] ),np.zeros([len(vv),300 ])
        fin_label=[]
        for n_case in range(len(cases_all)):
            filefullpath = os.path.join(os.path.curdir, DB_path[ch], cases_all[n_case])
            print(filefullpath)

            cases_all_p = [f for f in os.listdir(filefullpath) if ~os.path.isfile(os.path.join(filefullpath, f))]
            fin_label.extend(np.zeros(len(cases_all_p))+n_case)
            for n_case_p in range(len(cases_all_p)):

                path = os.path.join(os.path.curdir, DB_path[ch], cases_all[n_case], cases_all_p[n_case_p])
                print(path)
                im = cv2.imread(path)
                # resize the image #
                im  = cv2.resize(im, (cv_size,cv_size))
                plt.figure(1)
                plt.imshow(im)
                # --------<<<<<<<<<<STEP1:Pre-Processing  >>>>>>>>>>>>-------------#
                imm_c = cv2.medianBlur(im, 3)
                plt.figure(2)
                plt.imshow(imm_c)

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

                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im[:,:,0])
                vv=120
                x0 = int(maxLoc[0]) - vv
                y0 = int(maxLoc[1]) - vv
                x1 = int(maxLoc[0]) + vv
                y1 = int(maxLoc[1]) + vv
                roi_im = im[y0:y1, x0:x1]
                imm = Imedian_filter(roi_im)
                plt.figure(3)
                plt.imshow(imm)

                # --------<<<<<<<<<<STEP2:Feature Extraction >>>>>>>>>>>>-------------#

                # step2.1: Local Gabor XOR Pattern #
                from LGXP.LGXP import get_LGXP
                f1 = get_LGXP(imm)

                # step2.2: shape feature: improved hierarchy of skeleton #
                from HOS import hierarchy_skeletonize_image, hierarchyofskeleton_prop
                f2_c = hierarchy_skeletonize_image(imm)
                f2 = hierarchyofskeleton_prop(imm)

                # step2.3: deep feature #
                def deep_feat(input):
                    ##VGG16
                    # from keras.applications import VGG16
                    from keras.preprocessing import image
                    # from keras.applications.vgg16 import preprocess_input
                    # VGG_model = VGG16(weights='imagenet', include_top=False)
                    imgg = np.resize(input, (224, 224, 3))
                    img = image.img_to_array(imgg)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    features = VGG_model.predict(img)
                    vgg16_feat = np.histogram(features, bins=50)[0].reshape(-1, 1).flatten()

                    # Load the pre-trained ResNet50 model without the top (classification) layers
                    features_resnet = Resnet_model.predict(img)
                    Resnet_feat = np.histogram(features_resnet, bins=50)[0].reshape(-1, 1).flatten()

                    # Load the pre-trained Inception model without the top (classification) layers
                    features_inception = Inceptionv3_model.predict(img)
                    inception_feat = np.histogram(features_inception, bins=50)[0].reshape(-1, 1).flatten()

                    # Extract features from alexnet image
                    # AlexNet_feat = model_alexnet.predict(img)
                    # AlexNet_feat = np.histogram(AlexNet_feat, bins=50)[0].reshape(-1, 1).flatten()
                    deef_f = np.concatenate((vgg16_feat, Resnet_feat,inception_feat))
                    return deef_f
                f3 = deep_feat(imm)

                ff = np.concatenate((f1, f2,f3))
                ff_c = np.concatenate((f1, f2_c,f3))
                fin_feat[ii] =ff
                fin_feat_c[ii] =ff_c
                ii+=1
        if rn==1:
            np.save('saved data/fin_feat',fin_feat)
            np.save('saved data/fin_feat_c',fin_feat_c)
            np.save('saved data/fin_label',fin_label)

    if rn==1: read_feat_extract()
    db=1
    feat, label = load_data()

    class FireModule(object):
        """
        Fire Module computed as per the SqueezeNet paper
        """

        def __init__(self, layer_number: int, activation: str, kernel_initializer: str) -> None:
            """
            Constructor

            Arguments:
              layer_number       : Index of the Fire Module
              activation         : Activation to be used
              kernel_initializer : Kernel Weight Initialization technique

            Returns:
              None
            """

            self.layer_number = layer_number
            self.activation = activation
            self.kernel_initializer = kernel_initializer

        def build_module(self, fire_input: Layer) -> Layer:
            """
            Build the SqueezeNet

            Arguments:
              fire_input       : Input to Fire Module

            Returns:
              model            : SqueezeNet
            """

            global one, three, five

            output_size = 128 * (1 + (self.layer_number // 2))

            squeeze_1x1_filters = 16 * (1 + (self.layer_number // 2))
            expand_1x1_filters = expand_3x3_filters = output_size // 2

            squeeze_1x1 = Conv2D(name=f'fire_{self.layer_number + 2}_squeeze_1x1',
                                 filters=squeeze_1x1_filters, kernel_size=one, strides=1, padding='valid',
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer)(fire_input)
            expand_1x1 = Conv2D(name=f'fire_{self.layer_number + 2}_expand_1x1',
                                filters=expand_1x1_filters, kernel_size=one, strides=1, padding='valid',
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer)(squeeze_1x1)
            expand_3x3 = Conv2D(name=f'fire_{self.layer_number + 2}_expand_3x3',
                                filters=expand_3x3_filters, kernel_size=three, strides=1, padding='same',
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer)(squeeze_1x1)

            fire = Concatenate(name=f'fire_{self.layer_number + 2}')([expand_1x1, expand_3x3])

            return fire

    class SqueezeNet(object):
        """
        SqueezeNet Architecture
        """

        def __init__(self, activation: str = 'relu', kernel_initializer: str = 'glorot_uniform') -> None:
            """
            Constructor

            Arguments:
              activation         : Activation to be used
              kernel_initializer : Kernel Weight Initialization technique

            Returns:
              None
            """

            self.activation = activation
            self.kernel_initializer = kernel_initializer

        def vanilla_model(self, input_shape: tuple = (32, 32, 3), n_classes: int = 2) -> None:
            """
            Vanilla Implementation of SqueezeNet

            Arguments:
              input_shape         : Input Shape of the images
              n_classes           : Number of output classes

            Returns:
              None
            """
            from tensorflow.keras.layers import Input
            inp = Input(shape=input_shape, name='Input')

            # Conv1 Layer
            conv_1 = Conv2D(name="Conv_1",
                            filters=96, kernel_size=seven, strides=2, padding='same', activation=self.activation,
                            kernel_initializer=self.kernel_initializer)(inp)
            maxpool_1 = MaxPool2D(name="MaxPool_1",
                                  pool_size=three, strides=2)(conv_1)

            # Fire 2-4
            fire_2 = FireModule(layer_number=0, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_1)
            fire_3 = FireModule(layer_number=1, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_2)
            fire_4 = FireModule(layer_number=2, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_3)

            # Max Pool after Fire4 Module
            maxpool_2 = MaxPool2D(name="MaxPool_2",
                                  pool_size=three, strides=2)(fire_4)

            # Fire 5-8
            fire_5 = FireModule(layer_number=3, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_2)
            fire_6 = FireModule(layer_number=4, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_5)
            fire_7 = FireModule(layer_number=5, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_6)
            fire_8 = FireModule(layer_number=6, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_7)

            # Max Pool after Fire8 Module
            maxpool_3 = MaxPool2D(name="MaxPool_3",
                                  pool_size=three, strides=2)(fire_8)

            fire_9 = FireModule(layer_number=7, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_3)

            # Dropout
            dropout = Dropout(0.5, name="Dropout")(fire_9)

            # Conv10 layer
            conv_10 = Conv2D(name="Conv_10",
                             filters=1000, kernel_size=one, strides=1, padding='valid', activation=self.activation,
                             kernel_initializer=self.kernel_initializer)(dropout)
            gap_11 = GlobalAveragePooling2D()(conv_10)

            # Add Dense(n_classes) and ouput == Dense layer
            out = Dense(n_classes, activation='softmax')(gap_11)

            self.model = Model(inputs=inp, outputs=out)

        def bypass_model(self, input_shape: tuple = (32, 32, 3), n_classes: int = 2) -> None:
            """
            Residual Inspired Bypass Implementation of SqueezeNet

            Arguments:
              input_shape         : Input Shape of the images
              n_classes           : Number of output classes

            Returns:
              None
            """

            inp = Input(shape=input_shape, name='Input')

            # Conv1 Layer
            conv_1 = Conv2D(name="Conv_1",
                            filters=96, kernel_size=seven, strides=2, padding='same', activation=self.activation,
                            kernel_initializer=self.kernel_initializer)(inp)
            maxpool_1 = MaxPool2D(name="MaxPool_1",
                                  pool_size=three, strides=2)(conv_1)

            # Fire 2-4
            fire_2 = FireModule(layer_number=0, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_1)
            fire_3 = FireModule(layer_number=1, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_2)
            bypass_1 = Add(name="Bypass_1")([fire_2, fire_3])
            fire_4 = FireModule(layer_number=2, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(bypass_1)

            # Max Pool after Fire4 Module
            maxpool_2 = MaxPool2D(name="MaxPool_2",
                                  pool_size=three, strides=2)(fire_4)

            # Fire 5-8
            fire_5 = FireModule(layer_number=3, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_2)
            bypass_2 = Add(name="Bypass_2")([maxpool_2, fire_5])
            fire_6 = FireModule(layer_number=4, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(bypass_2)
            fire_7 = FireModule(layer_number=5, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(fire_6)
            bypass_3 = Add(name="Bypass_3")([fire_6, fire_7])
            fire_8 = FireModule(layer_number=6, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(bypass_3)

            # Max Pool after Fire8 Module
            maxpool_3 = MaxPool2D(name="MaxPool_3",
                                  pool_size=three, strides=2)(fire_8)

            fire_9 = FireModule(layer_number=7, activation=self.activation,
                                kernel_initializer=self.kernel_initializer).build_module(maxpool_3)
            bypass_4 = Add(name="Bypass_4")([maxpool_3, fire_9])

            # Dropout
            dropout = Dropout(0.5, name="Dropout")(bypass_4)

            # Conv10 layer
            conv_10 = Conv2D(name="Conv_10",
                             filters=1000, kernel_size=one, strides=1, padding='valid', activation=self.activation,
                             kernel_initializer=self.kernel_initializer)(dropout)
            gap_11 = GlobalAveragePooling2D()(conv_10)

            out = Dense(4, activation='softmax')(gap_11)

            self.model = Model(inputs=inp, outputs=out)

        def build_model(self, input_shape: tuple = (32, 32, 3), n_classes: int = 4, choice: str = 'vanilla') -> Model:
            """
            Build SqueezeNet

            Arguments:
              input_shape         : Input Shape of the images
              n_classes           : Number of output classes
              choice              : Type of architecture (vanilla/bypass)
            Returns:
              model               : SqueezeNet Model
            """

            if choice == "vanilla":
                self.vanilla_model(input_shape, n_classes)
            else:
                self.bypass_model(input_shape, n_classes)

            return self.model

    def csqueezenet_(X_train, Y_train, X_test, Y_test, db):
        tr_data, tr_lab, tst_data, tst_lab = X_train, Y_train, X_test, Y_test
        target = np.concatenate((tr_lab, tst_lab), axis=0)
        ln = len(set(Y_train.flatten()))
        xx = 500
        snet = SqueezeNet()
        model = snet.build_model(n_classes=ln, choice='vanilla')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model.summary()
        tr_lab_init = tr_lab
        tr_lab = tf.keras.utils.to_categorical(tr_lab)
        tr_data = np.resize(tr_data, (xx, 32, 32, 3))
        tr_lab = np.resize(tr_lab, (xx, ln))
        from tensorflow.keras.models import load_model
        if rn == 1: model.fit(tr_data, tr_lab, batch_size=64, epochs=ep,
                              verbose=0)  # model.save('saved data/squeezenet_model.h5')
        model = load_model('saved data/squeezenet_model.h5')
        # plot_model(model, to_file='model_squeezenet.jpg', show_shapes=True, dpi=800)
        tst_data = np.resize(tst_data, (X_test.shape[0], 32, 32, 3))
        y_pred = model.predict(tst_data)
        y_pred = y_pred.argmax(axis=-1)
        # y_pred = np.resize(y_pred, (tr_lab_init.shape))
        y_pred = array(y_pred, axis=[2, db])
        out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
        return out, y_pred, cm[0], cm[1]

    def Icnn_net(X_train, Y_train, X_test, Y_test, db):
        ln = len(set(Y_train.flatten()))
        cnn_X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
        cnn_X_test = X_test.reshape((X_test.shape[0], 1, 1, X_test.shape[1]))
        y_train = to_categorical(Y_train)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(1, 1), strides=(2, 2), padding='valid', input_shape=cnn_X_train[0].shape))
        model.add(LeakyReLU(alpha=0.01))  # You can adjust the alpha parameter
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(1, 1),  strides=(2, 2),padding='valid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        model.add(Conv2D(128, kernel_size=(1, 1), strides=(2, 2), padding='valid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        model.add(Conv2D(256, kernel_size=(1, 1), strides=(2, 2), padding='valid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        # model.add(Conv2D(256, kernel_size=(1, 1),  strides=(2, 2),padding='valid'))
        # model.add(LeakyReLU(alpha=0.01))
        # model.add(BatchNormalization())
        #
        # model.add(Conv2D(256, kernel_size=(1, 1), strides=(2, 2),padding='valid'))
        # model.add(LeakyReLU(alpha=0.01))
        # model.add(BatchNormalization())

        model.add(Conv2D(512, kernel_size=(1, 1), strides=(2, 2), padding='valid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Conv2D(2, kernel_size=(1, 1),  strides=(2, 2),padding='valid'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())

        # dense layers (for classification)
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        def update_activation(x):
            from keras import backend as K
            import tensorflow as tf, sys, math
            """comb H sine HT activation function.
    
            # Arguments
                x: Input tensor.
                axis: Integer, axis along which the comb_H_sine_HT normalization is applied.
    
            # Returns
                Tensor, output of softmax transformation.
    
            # Raises
                ValueError: In case `dim(x) == 1`.
            """
            ndim = K.ndim(x)
            if ndim == 2:
                return K.softmax(x)
            else:
                def _to_tensor(x, dtype):
                    """Convert the input `x` to a tensor of type `dtype`.

                    # Arguments
                        x: An object to be converted (numpy array, list, tensors).
                        dtype: The destination type.

                    # Returns
                        A tensor.
                    """
                    return tf.convert_to_tensor(x, dtype=dtype)

                t1 = tf.convert_to_tensor(0.01, dtype=tf.float32)
                t2 = tf.convert_to_tensor(2, dtype=tf.float32)

                # Use tf.cond to conditionally execute the subgraphs
                tensor_shape, X = tf.shape(x), int(x.shape[1])
                # def sinusoidal():
                #     bn = np.random.uniform(0, 1)
                #     bn_1 = 2.3 * bn ** 2 * math.sin(math.pi * bn)  # math.cos(0.5 * math.acos(ck))
                #     return bn_1
                Bn = 0.5
                comb_h_sin = math.sinh(Bn * X) + math.asinh(Bn * X)
                # hyperbolic tangent #
                X = tf.convert_to_tensor(X, dtype=tf.float32)

                # HT_x = (2 / (1 + math.exp(-X))) -1
                HT_x = ((2*K.exp(x)  + K.exp(2*x)-1 ) / (K.exp(x) +K.exp(-x) +K.exp(-2*x)+1)) + comb_h_sin#(2 / (1 + K.exp(-X))) - 1
                comb_h_sin = tf.convert_to_tensor(comb_h_sin, dtype=tf.float32)
                comb_h_sin_HT = (1/2) * (comb_h_sin + HT_x)
                sys.stdout.write(str(comb_h_sin_HT))
                return comb_h_sin_HT

        model.add(Dense(ln, activation=update_activation))
        # model.add(Dense(ln, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(cnn_X_train, Y_train, epochs=ep, batch_size=64, verbose=0)
        print(model.summary())
        y_pred = model.predict(cnn_X_test)
        y_pred = array(np.argmax(y_pred, axis=1), axis=[1, db])
        out, cm = multi_confu_matrix(Y_test, y_pred, l, m)
        return out, y_pred, cm[0], cm[1]

    # --------<<<<<<<<<<STEP3:Classification >>>>>>>>>>>>-------------#

    def prop_lb(X_train, X_test,Y_train ,Y_test, db):
            # --------<<<<<<<<<<STEP4:Detection >>>>>>>>>>>>-------------#
            print('squeezenet Model \n')
            out1, y_pred1, ttn1, cm1 = csqueezenet_(X_train, Y_train, X_test, Y_test, db)
            print('improved CNN Model\n')
            out2, y_pred2,ttn2, cm2 = Icnn_net(X_train, Y_train, X_test, Y_test, db)
            y_predict = array(np.round(np.mean((y_pred1, y_pred2), axis=0)),axis=[1,db])
            return y_predict

    def Testing_process(feat, label, db):
            def comp(X_train, X_test, Y_train, Y_test, db):
                pred = [lstm_, dcnn_, bigru_,  csqueezenet_, dense_net, bp1_, bp2_]
                pos = len(pred)
                lp = np.load('pre_evaluated/tp.npy')
                pred_out = []
                for i in range(0, pos):
                    print('lp=' + str(lp) + '_method=' + str(i))
                    out, y_pred, cm, ttn = pred[i](X_train, Y_train, X_test, Y_test, db)
                    pred_out.append(y_pred)
                return pred_out

            def metrices_(pred, Y_test):
                out = multi_confu_matrix(Y_test, pred)
                return out

            def method(X_train, X_test, Y_train, Y_test):
                vv = []
                vv.extend(comp(X_train, X_test, Y_train, Y_test, db))
                vv.append(prop_lb(X_train, X_test, Y_train, Y_test, db))
                return vv

            learn_percent, learning_percentage = [0.6, 0.7, 0.8, 0.9], ['60', '70', '80', '90']
            for lp, lpstr in zip(learn_percent, learning_percentage):
                feat = feat / np.max(feat)
                X_train, X_test, Y_train, Y_test = train_test_split(feat, label, train_size=lp, random_state=0)
                np.save(f'pre_evaluated/Actual-' + str(lpstr)+str(db), Y_test)
                np.save('pre_evaluated/Y_test', Y_test)
                np.save('pre_evaluated/tp', lp)
                pred_va = np.array(method(X_train, X_test, Y_train, Y_test))
                np.save('pre_evaluated/Predicted-'  + str(lpstr)+str(db), pred_va)
                #out = np.array([metrices_(pre, Y_test) for idx, pre in enumerate(pred_va)])
                #result_ = np.array([out[i][0] for i in range(len(out))])
                #TPTN = np.array([out[i][1][0] for i in range(len(out))])
                #CM = np.array([out[i][1][1] for i in range(len(out))])

                from sklearn.metrics import ConfusionMatrixDisplay
                import matplotlib.pyplot as plt

                # Calculate metrics
                out = np.array(
                    [metrices_(pre, Y_test) for idx, pre in enumerate(pred_va)],
                    dtype=object
                )

                # Extract results
                result_ = np.array([out[i][0] for i in range(len(out))], dtype=object)
                TPTN = np.array([out[i][1][0] for i in range(len(out))], dtype=object)
                CM = np.array([out[i][1][1] for i in range(len(out))], dtype=object)

                # Model names
                clmn = [
                    'LSTM',
                    'DCNN',
                    'BIGRU',
                    'SqueezeNet',
                    'DenseNet',
                    'BASE1',
                    'BASE2',
                    'PROPOSED'
                ]

                print("Number of confusion matrices:", len(CM))

                # Plot confusion matrices
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                for i, ax in enumerate(axes.flatten()):

                    if i < len(CM):
                        print("\nModel:", clmn[i])
                        print("CM raw =", CM[i])
                        print("CM type =", type(CM[i]))

                        cm = np.array(CM[i], dtype=float)

                        print("CM shape =", cm.shape)

                        # Normalize confusion matrix
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm_norm
                        )

                        disp.plot(
                            ax=ax,
                            cmap='Greens',
                            values_format='.2f',
                            colorbar=False
                        )

                        ax.set_title(clmn[i])

                plt.tight_layout()
                plt.show()



                # lstm_, dcnn_, bigru_,  csqueezenet_, dense_net, bp1_, bp2_
                clmn = ['LSTM', 'DCNN', 'BIGRU', 'SqueezeNet', 'DenseNet','BASE1', 'BASE2', 'PROPOSED']
                indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
                globals()['df' + lpstr] = pd.DataFrame(result_.transpose(), columns=clmn, index=indx)
                indx1 = ['TP', 'TN', 'FP', 'FN']
                globals()['df_' + lpstr] = pd.DataFrame(TPTN.transpose(), columns=clmn, index=indx1)
                np.save('pre_evaluated/cm-' + str(lpstr)+str(db), CM)
            key = ['60', '70', '80', '90']
            frames = [df_60, df_70, df_80, df_90]
            df1 = pd.concat(frames, keys=key, axis=0)
            df1.to_csv(f'pre_evaluated/TP_TN-'+ str(db)+ '.csv')
            key = ['60', '70', '80', '90']
            frames = [df60, df70, df80, df90]
            df1 = pd.concat(frames, keys=key, axis=0)
            df1.to_csv(f'pre_evaluated/Performance-' +str(db) +'.csv')

    Testing_process(feat, label, db)

    def prop_lb_clf(X_train, X_test,Y_train ,Y_test, db):
            # --------<<<<<<<<<<STEP4:Detection >>>>>>>>>>>>-------------#
            print('squeezenet Model \n')
            out1, y_pred1, ttn1, cm1 = csqueezenet_(X_train, Y_train, X_test, Y_test, db)
            print('CNN Model\n')
            out2, y_pred2,ttn2, cm2 = cnn_net(X_train, Y_train, X_test, Y_test, db)
            y_predict = array(np.round(np.mean((y_pred1, y_pred2), axis=0)),axis=[1,db])
            return y_predict

    def ab_comp():
        feat_o = np.load('saved data/fin_feat' + '.npy')
        feat1 = np.load('saved data/fin_feat_c' + '.npy', allow_pickle=True)
        label = np.load('saved data/fin_label' + '.npy')
        f, f_ = [feat_o, feat1],'c'

        label1 = [label, label]
        lp1 = [0.7]
        met1, lab1, cm1 = [], [], []
        for lp in lp1:
            c1 = []
            for i in range(0, len(label1)):
                print('LP >>>>>', lp,i)
                fet = f[i]
                fet[np.where(np.isnan(fet))] = 0.1
                fet1 = fet / np.max(fet)
                X_train, X_test, Y_train, Y_test = train_test_split(fet1, label1[i][:len(fet1)], test_size=0.33, random_state=0)
                np.save('pre_evaluated/Y_test', Y_test)
                np.save('pre_evaluated/tp', lp)
                vv =prop_lb(X_train, X_test, Y_train, Y_test, db)
                c1.append( multi_confu_matrix(Y_test, vv))
            vv = prop_lb_clf(X_train, X_test, Y_train, Y_test, db)
            c1.append(multi_confu_matrix(Y_test, vv))
            comp=np.array(c1)
            vall, clasify_roc, cmm = [], [], []
            for jj in range(len(comp)):
                vall.append(comp[jj][0])
                cmm.append(comp[jj][1])
            class_val, roc_comp = [], []
            for kk in range(len(vall)):
                class_val.append(vall[kk])
            class_val = np.array(class_val)
            met1.append(vall)
            lab1.append(class_val)
            cm1.append(cmm)
        out_com = np.array(met1)[0,:,:]*hr()
        clmn = ['conventional preprocess','conventional hierarchy of skeleton','squeezenet+CNN']
        indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
        import pandas as pd
        val = np.array(out_com)
        d = pd.DataFrame(val.transpose(), columns=clmn, index=indx)
        if rn == 1: d.to_csv(f'pre_evaluated/ablation_study' + str(db)+'.csv')
        return d

    ab_comp()


###### main execution ##############
tt=0
vvv = popup.PopupYesNo('')
if vvv == "Full analysis + Plots":
    full_analysis()
    plotresult()
    plt.show()
else:
    plotresult()
    plt.show()


# # full_analysis()
# plotresult()
# plt.show()
