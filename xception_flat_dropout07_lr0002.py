import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Model,model_from_json,Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, \
    BatchNormalization, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from sklearn.model_selection import train_test_split
import os
from os.path import dirname, abspath
from contextlib import redirect_stdout
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K


# all Convlution and Separable Convolution layers are followed by batch normalization
class XceptionModel:
    def __init__(self,dict, dropout_rate):
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
        self.dropout_rate = dropout_rate

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

        #TESTING
        #x = MaxPooling2D(3, strides=2, padding='same')(x)

        # or we can use Flatten() instead.
        x = Flatten()(x)
        #x = GlobalAveragePooling2D()(x)
        #x = GlobalMaxPooling2D()(x)
        # outputs probability that the video will be FAKE
        x = Dropout(self.dropout_rate)(x)
        '''
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Activation('relu')(x)
        '''
        x = Dense(1, activation='sigmoid')(x)

        if DEBUG:
            print(x.shape)

        return x

    def forward(self,input):
        x = self.entry_flow(input)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.batch_loss = []
        self.batch_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_acc.append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

def train_gen(batch_size, images, labels, data_dir):
    batch_start = 0
    batch_end = batch_size
    while True:
        train_examples = []
        limit = min(batch_end, len(images))
        for frame in images[batch_start:limit]:
            video_array = np.load(data_dir + '/' + frame + '.npy')
            train_examples.append(np.array(video_array))

        yield np.array(train_examples), labels[batch_start:limit]

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size


def valid_gen(batch_size, images, labels, data_dir):
    batch_start = 0
    batch_end = batch_size
    while True:
        train_examples = []
        limit = min(batch_end, len(images))
        for frame in images[batch_start:limit]:
            video_array = np.load(data_dir + '/' + frame + '.npy')
            train_examples.append(np.array(video_array))

        yield np.array(train_examples), labels[batch_start:limit]

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size


def test_gen(batch_size, images, labels, data_dir):
    batch_start = 0
    batch_end = batch_size
    while True:
        train_examples = []
        limit = min(batch_end, len(images))
        for frame in images[batch_start:limit]:
            video_array = np.load(data_dir + '/' + frame + '.npy')
            train_examples.append(np.array(video_array))

        yield np.array(train_examples), labels[batch_start:limit]

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size

def main():
    batch_size = 8 # hyperparameter
    epochs = 20 # hyperparameter
    dropout_rate = 0.7
    validation_split = 0.1
    test_split = 0.05
    class_weights = {0: 4,
                     1: 0.6}

    homedir = dirname(dirname(abspath(__file__)))
    data_dir = homedir + '/output/stacked_frames/'
    datatable_file = 'dataframe.csv'
    output_dir = homedir + '/output/final_Xception_training/top_n_results_weighted_lr_decr/params_5_flat_dropout07_lr0002/'
    params_dict_folder = homedir + '/output/final_Xception_training/top_n_params/'
    params_dict_files = [params_dict_folder + x for x in os.listdir(params_dict_folder)]

    data = pd.read_csv(data_dir + '/' + datatable_file)
    data = data.loc[data['num_empty_frames'] < 5]

    images = data['index'].tolist()
    labels = data['fake'].tolist()

    n_pos = len([x for x in labels if x == 0])
    n_neg = len(labels) - n_pos
    print("positive num: " + str(n_pos) + ', negative num: ' + str(n_neg))

    input_shape = np.load(data_dir + '/' + images[0] + '.npy').shape


    parameters = []
    for file in params_dict_files:
        param_dict = {}
        with open(file, 'r') as f:
            for line in f:
                p, val = line.replace(":", "").split()
                param_dict[p] = int(val)
        parameters.append(param_dict.copy())

    parameter = parameters[5]

    model = XceptionModel(parameter, dropout_rate)
    inputs = Input(shape=input_shape)
    outputs = model.forward(inputs)
    model = Model(inputs, outputs)

    adam = Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    history = LossHistory()

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_split, random_state=42)

    num_train_examples = int(np.ceil(len(x_train) * (1 - validation_split)))
    num_valid_examples = len(x_train) - num_train_examples
    print('\nnum of train examples: ' + str(num_train_examples))
    print('num of valid examples: ' + str(num_valid_examples))
    print('steps_per_epoch: ' + str(int(np.ceil(num_train_examples / batch_size))))
    print('validation_steps: ' + str(int(np.ceil(num_valid_examples / batch_size))) + '\n')

    #images, labels = shuffle(images, labels, random_state=0)
    train_generator = train_gen(batch_size, x_train[:num_train_examples], y_train[:num_train_examples], data_dir)
    val_generator = valid_gen(batch_size, x_train[num_train_examples:], y_train[num_train_examples:], data_dir)

    print('starting training')

    # with generator
    model.fit_generator(train_generator,
                        steps_per_epoch=int(np.ceil(num_train_examples / batch_size)),
                        validation_data=val_generator,
                        validation_steps=int(np.ceil(num_valid_examples / batch_size)),
                        class_weight=class_weights,
                        callbacks=[history],
                        epochs=epochs)

    test_generator = test_gen(batch_size, x_test, y_test, data_dir)

    num_batches = int(np.ceil(len(x_test) / batch_size))

    predictions = model.predict_generator(test_generator, steps=num_batches)
    predictions = np.array(predictions).flatten()

    test_error = np.array(y_test) - predictions
    df = pd.DataFrame(data={'filename': x_test[:], 'error': test_error[:]})

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    model.save(output_dir + 'Xception.h5')
    print('Model saved.')

    df.to_csv(output_dir + 'test_error_table.csv', index=False)

    with open(output_dir + 'train_params.txt', 'w+') as f:
        for attr, value in parameter.items():
            f.write(str(attr) + ': ' + str(value) + '\n')

    with open(output_dir + 'model_summary.txt', 'w+') as f:
        with redirect_stdout(f):
            model.summary()

    np.savetxt(output_dir + 'history_loss.txt', np.array(history.loss))
    np.savetxt(output_dir + 'history_acc.txt', np.array(history.acc))
    np.savetxt(output_dir + 'history_batch_loss.txt', np.array(history.batch_loss))
    np.savetxt(output_dir + 'history_batch_acc.txt', np.array(history.batch_acc))
    np.savetxt(output_dir + 'history_val_acc.txt', np.array(history.val_acc))
    np.savetxt(output_dir + 'history_val_loss.txt', np.array(history.val_loss))

if __name__ == "__main__":
    main()