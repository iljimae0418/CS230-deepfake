import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models,layers
from tensorflow.keras.models import Model,model_from_json,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# all Convlution and Separable Convolution layers are followed by batch normalization
class XceptionModel:
    def __init__(self,dict):
        ''' entry flow hyperparameters '''
        self.firstConv_filters = dict["firstConv_filters"] # number of filters
        self.firstConv_filterSize = dict["firstConv_filterSize"] # size of filters
        self.firstConv_filterStride = dict["firstConv_filterStride"] # stride of filters

        self.secondConv_filters = dict["secondConv_filters"]
        self.secondConv_filterSize = dict["secondConv_filterSize"]
        self.secondConv_filterStride = dict["secondConv_filterStride"]

        self.entry_residual_blocks = dict["entry_residual_blocks"]
        self.entry_residual_filters = dict["entry_residual_filters"]
        self.entry_residual_filterSize = dict["entry_residual_filterSize"]
        self.entry_residual_filterStride = dict["entry_residual_filterStride"]

        ''' middle flow hyperparameters '''
        self.middle_residual_blocks = dict["middle_residual_blocks"]
        self.middle_residual_filters = dict["middle_residual_filters"]
        self.middle_residual_filterSize = dict["middle_residual_filterSize"]
        self.middle_residual_filterStride = dict["middle_residual_filterStride"]

        ''' exit flow hyperparameters '''
        self.exit_residual_blocks = dict["exit_residual_blocks"]
        self.exit_residual_filters1 = dict["exit_residual_filters1"]
        self.exit_residual_filterSize1 = dict["exit_residual_filterSize1"]
        self.exit_residual_filterStride1 = dict["exit_residual_filterStride1"]

        self.exit_residual_filters2 = dict["exit_residual_filters2"]
        self.exit_residual_filterSize2 = dict["exit_residual_filterSize2"]
        self.exit_residual_filterStride2 = dict["exit_residual_filterStride2"]

        self.exit_filters1 = dict["exit_filters1"]
        self.exit_filterSize1 = dict["exit_filterSize1"]
        self.exit_filterStride1 = dict["exit_filterStride1"]

        self.exit_filters2 = dict["exit_filters2"]
        self.exit_filterSize2 = dict["exit_filterSize2"]
        self.exit_filterStride2 = dict["exit_filterStride2"]

        self.fully_connected_flow_layers = dict["fully_connected_flow_layers"]

    ''' the entry flow similar to that described in the architecture diagram '''
    def entry_flow(self,inputs,DEBUG=True):
        # entry convolutional layers
        x = Conv2D(self.firstConv_filters,self.firstConv_filterSize,
                    strides=self.firstConv_filterStride,padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(self.secondConv_filters,self.secondConv_filterSize,
                    strides=self.secondConv_filterStride,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        previous_block_activation = x

        for _ in range(self.entry_residual_blocks):
            x = Activation('relu')(x)
            x = SeparableConv2D(self.entry_residual_filters,self.entry_residual_filterSize,
                                strides=self.entry_residual_filterStride,padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(self.entry_residual_filters,self.entry_residual_filterSize,
                                strides=self.entry_residual_filterStride,padding='same')(x)
            x = BatchNormalization()(x)

            # max pooling layer that we may potentially get rid of
            x = MaxPooling2D(3,strides=2,padding='same')(x)

            # the residual connection as described in the architecture diagram
            residual = Conv2D(self.entry_residual_filters,1,strides=2,padding='same')(previous_block_activation)
            x = Add()([x,residual])
            previous_block_activation = x

        if DEBUG:
            print(x.shape)
        return x

    ''' the middle flow similar to that described in the architecture diagram '''
    def middle_flow(self,x,DEBUG=True):
        previous_block_activation = x
        for _ in range(self.middle_residual_blocks):
            x = Activation('relu')(x)
            x = SeparableConv2D(self.middle_residual_filters,self.middle_residual_filterSize,
                                strides=self.middle_residual_filterStride,padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(self.middle_residual_filters,self.middle_residual_filterSize,
                                strides=self.middle_residual_filterStride,padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(self.middle_residual_filters,self.middle_residual_filterSize,
                                strides=self.middle_residual_filterStride,padding='same')(x)
            x = BatchNormalization()(x)

            # skip connection
            x = Add()([x,previous_block_activation])
            previous_block_activation = x

        if DEBUG:
            print(x.shape)
        return x

    ''' the exit flow similar to that descrbed in the architecture diagram '''
    def exit_flow(self,x,DEBUG=True):
        previous_block_activation = x
        for _ in range(self.exit_residual_blocks):
            x = Activation('relu')(x)
            x = SeparableConv2D(self.exit_residual_filters1,self.exit_residual_filterSize1,
                                strides=self.exit_residual_filterStride1,padding='same')(x)
            x = BatchNormalization()(x)

            x = Activation('relu')(x)
            x = SeparableConv2D(self.exit_residual_filters2,self.exit_residual_filterSize2,
                                strides=self.exit_residual_filterStride2,padding='same')(x)
            x = BatchNormalization()(x)

            # we may get rid of this max pooling layer
            x = MaxPooling2D(3,strides=2,padding='same')(x)

            # skip connection with Conv2D
            residual = Conv2D(self.exit_residual_filters2,1,strides=2,padding='same')(previous_block_activation)
            x = Add()([x,residual])
            previous_block_activation = x

        x = Activation('relu')(x)
        x = SeparableConv2D(self.exit_filters1,self.exit_filterSize1,
                            strides=self.exit_filterStride1,padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(self.exit_filters2,self.exit_filterSize2,
                            strides=self.exit_filterStride2,padding='same')(x)
        x = BatchNormalization()(x)

        # or we can use Flatten() instead.
        x = GlobalAveragePooling2D()(x)
        # outputs probability that the video will be FAKE


    def fully_connected_res_flow(self, x):
        tempnodes = x.get_shape().as_list()
        temp = Dense(tempnodes[1])(x)
        temp = Activation(activation="selu")(temp)
        temp = BatchNormalization()(temp)
        temp2 = Add()([x, temp])
        return temp2

    def fully_connected_flow(self,x,DEBUG=True):

        previous_exit_flow = x
        for _ in range(self.fully_connected_flow_layers):
            previous_exit_flow = self.fully_connected_flow(previous_exit_flow,DEBUG=True)

        previous_exit_flow = Dense(1,activation='linear')(previous_exit_flow)

        if DEBUG:
            print(previous_exit_flow.shape)

        return previous_exit_flow

    def forward(self,input):
        x = self.entry_flow(input)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.fully_connected_flow(x)
        return x

''' example '''
# X_train,Y_train,X_val,Y_val,X_test,Y_test prepared beforehand
# check dimensions when using parameters
parameters = {"firstConv_filters":32,"firstConv_filterSize":3,"firstConv_filterStride":2,
                "secondConv_filters":64,"secondConv_filterSize":3,"secondConv_filterStride":1,
                "entry_residual_blocks":10,"entry_residual_filters":128,"entry_residual_filterSize":10,"entry_residual_filterStride":1,
                "middle_residual_blocks":10,"middle_residual_filters":128,"middle_residual_filterSize":3,"middle_residual_filterStride":1,
                "exit_residual_blocks":10,"exit_residual_filters1":128,"exit_residual_filterSize1":3,"exit_residual_filterStride1":1,
                "exit_residual_filters2":1024,"exit_residual_filterSize2":3,"exit_residual_filterStride2":1,
                "exit_filters1":728,"exit_filterSize1":3,"exit_filterStride1":1,
                "exit_filters2":1024,"exit_filterSize2":3,"exit_filterStride2":1}
model = XceptionModel(parameters)
width,height,depth = 244,244,30 # hyperparameter
inputs = Input(shape=(width,height,depth))
outputs = model.forward(inputs)
xception = Model(inputs,outputs)
xception.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
batch_size = 512 # hyperparameter
epochs = 1000 # hyperparameter
''' the lines below can be used for trainig if train and test data is prepared '''
# history = xception.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_val,Y_val))
# predicted = xception.predict(X_test)

