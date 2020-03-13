from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import regularizers, initializers
import keras
from contextlib import redirect_stdout
#from sklearn.utils import shuffle

import torch
import os
from os.path import dirname, abspath
import numpy as np
import pandas as pd
import keras.backend as K

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.batch_loss = []
        self.batch_acc = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_acc.append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

def processImage(image):
    image = image.astype('float64')
    image = image/255.0
    return image


def train_ex_Gen(batch_size, images, labels, dir):
    batch_start = 0
    batch_end = batch_size
    while True:
        train_examples = []
        limit = min(batch_end, len(images))
        for stacked_frames in images[batch_start:limit]:
            video_tensor = np.load(dir + '/' + stacked_frames + '.npy')
            train_examples.append(np.array(video_tensor))

        yield np.array(train_examples), labels[batch_start:limit]

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size


def valid_ex_Gen(batch_size, images, labels, dir):
    batch_start = 0
    batch_end = batch_size
    while True:
        train_examples = []
        limit = min(batch_end, len(images))
        for stacked_frames in images[batch_start:limit]:
            video_tensor = np.load(dir + '/' + stacked_frames + '.npy')
            train_examples.append(np.array(video_tensor))

        yield np.array(train_examples), labels[batch_start:limit]

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size


def lrcn(input_shape):
    def add_default_block(model, kernel_filters):
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same')))
        model.add(TimeDistributed(Activation('relu', name='activation_-2')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout(0.5)))
        # conv
        model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same')))
        model.add(TimeDistributed(Activation('relu', name='activation_-1')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout(0.5)))
        # max pool
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        return model

    reg_lambda = 0.001

    model = Sequential()

    # first (non-default) block
    model.add(TimeDistributed(Conv2D(16, (7, 7), strides=(2, 2), padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.3)))
    #model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    model = add_default_block(model, 64)
    model = add_default_block(model, 128)
    #model = add_default_block(model, 256)
    #model = add_default_block(model, 512)

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(1024, name='dense_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, name='pre-sigmoid'))
    model.add(Activation('sigmoid', name='final_activation'))

    return model


def custom_loss(y_true, y_pred):
    weights = (y_true * 1.6) + 0.4
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def ExtractFaces(video, data_dir):
    video_tensor = []
    images_list = os.listdir(data_dir)
    if video + "_empty" not in images_list:
        tlist = torch.load(data_dir + video + '.pt')
        for t in tlist:
            if isinstance(t, np.ndarray):
                video_tensor.append(np.transpose(t, (1, 2, 0)))
            else:
                video_tensor.append(t.permute(1, 2, 0).int().numpy())

    return np.array([video_tensor])

def train(video_paths, labels, model, data_dir, num_epochs):
    images_list = os.listdir(data_dir)
    for i in range(num_epochs):
        trial = 1
        for video_path, label in zip(video_paths, labels):
            if video_path + "_empty" not in images_list:
                print("Video #  {}...".format(trial))
                faces = ExtractFaces(video_path, data_dir)

                [loss, accuracy] = model.train_on_batch(faces, np.array([label]))
                print('loss is {} accuracy is {}'.format(loss, accuracy))

                trial += 1

    return model

def main():

    batch_size = 4
    num_epochs = 6
    validation_split = 0.15

    homedir = dirname(dirname(abspath(__file__)))
    data_dir = homedir + '/output/stacked_frames/'
    datatable_file = 'dataframe.csv'
    output_dir = homedir + '/output/LSTM_training/'

    for subdir, dirs, files in os.walk(data_dir):
        for dir in dirs:
            if dir != '244_m100_70frames':
                continue
            dir = data_dir + dir
            data = pd.read_csv(dir + '/' + datatable_file)[:600]
            data = data.loc[data['num_empty_frames'] < 5]

            images = data['index'].tolist()
            labels = data['fake'].tolist()

            input_shape = np.load(dir + '/' + images[0] + '.npy').shape

            model = lrcn(input_shape)
            # print(model.summary())

            adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
            #sgd = SGD(lr=0.0001, clipvalue=0.5)
            #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

            model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])

            history = LossHistory()

            num_train_examples = int(np.ceil(len(images) * (1 - validation_split)))
            num_valid_examples = len(images) - num_train_examples

            #images, labels = shuffle(images, labels, random_state=0)

            train_Gen = train_ex_Gen(batch_size, images[:num_train_examples], labels[:num_train_examples], dir)
            valid_Gen = valid_ex_Gen(batch_size, images[num_train_examples:], labels[num_train_examples:], dir)

            print('starting training')

            # with generator
            model.fit_generator(train_Gen,
                                steps_per_epoch=np.ceil(num_train_examples / batch_size),
                                validation_data=valid_Gen,
                                validation_steps=np.ceil(num_valid_examples / batch_size),
                                callbacks=[history],
                                epochs=num_epochs)

            output_dir_loc = output_dir + 'from_' + os.path.basename(dir) + '_arch2/'
            if not (os.path.exists(output_dir_loc)):
                os.makedirs(output_dir_loc)

            model.save(output_dir_loc + 'CNN_LSTM.h5')
            print('Model saved.')

            with open(output_dir_loc + 'model_summary.txt', 'w+') as f:
                with redirect_stdout(f):
                    model.summary()

            np.savetxt(output_dir_loc + 'history_loss.txt', np.array(history.loss))
            np.savetxt(output_dir_loc + 'history_acc', np.array(history.acc))
            np.savetxt(output_dir_loc + 'history_batch_loss.txt', np.array(history.batch_loss))
            np.savetxt(output_dir_loc + 'history_batch_acc.txt', np.array(history.batch_acc))
            np.savetxt(output_dir_loc + 'history_val_acc.txt', np.array(history.val_acc))

if __name__ == "__main__":
    main()
