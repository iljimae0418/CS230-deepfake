import torch
import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mplimg
from sklearn.utils import shuffle
from contextlib import redirect_stdout

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Conv2D, Activation
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras


#avg_images = output_dir + '/averaging/images/'


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




def train_ex_Gen(batch_size, images, labels):
    batch_start = 0
    batch_end = batch_size
    while True:
        limit = min(batch_end, len(images))

        train_examples = np.array(images[batch_start:limit])
        labels_batch = np.array(labels[batch_start:limit])
        yield train_examples, labels_batch

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size


def valid_ex_Gen(batch_size, images, labels):
    batch_start = 0
    batch_end = batch_size
    while True:
        limit = min(batch_end, len(images))

        yield np.array(images[batch_start:limit]), np.array(labels[batch_start:limit])

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(images):
            batch_start = 0
            batch_end = batch_size



def main():

    batch_size = 64
    num_epochs = 7
    validation_split = 0.15

    homedir = dirname(dirname(abspath(__file__)))
    output_dir = homedir + '/output2/'

    images = []
    labels = []
    for i, file in enumerate(os.listdir(output_dir + 'no_avg/')):
        if file.endswith(".pt"):
            tlist = torch.load(output_dir + 'no_avg/' + file)
            for t in tlist:
                images.append(t.permute(1, 2, 0).int().numpy())

    for file in os.listdir(output_dir + 'no_avg/'):
        if file.endswith(".txt"):
            labels.extend(np.loadtxt(output_dir + 'no_avg/' + file))

    print('training data loaded')

    n_pos = len([x for x in labels if x == 0])
    n_neg = len(labels) - n_pos

    input_shape = images[0].shape

    model = Sequential()

    model.add(Conv2D(8, (5, 5),
                     padding='same',
                     input_shape=input_shape,
                     strides=(1, 1),
                     data_format="channels_last",
                     name='conv_layer1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(16, (4, 4),
                     strides=(1, 1),
                     padding='same',
                     name='conv_layer3'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(1, 1),
                           padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     name='conv_layer5'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     name='conv_layer6'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(1, 1),
                           padding='same'))

    model.add(Flatten())

    model.add(Dense(120, name='dense_layer1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(84, name='attribute_layer'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1, name='pre-sigmoid'))
    model.add(Activation('sigmoid'))

    # initiate the Adam optimizer with a given learning rate (Note that this is adapted later)
    opt = keras.optimizers.Adam(lr=0.001)

    # Compile the model with the desired loss, optimizer, and metric
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if not (os.path.exists(output_dir + '/CNN_from_scratch/CNN2/')):
        os.makedirs(output_dir + '/CNN_from_scratch/CNN2/')

        write_dir = output_dir + '/CNN_from_scratch/CNN2/'

    with open(write_dir + 'CNN1_summary.txt', 'w+') as f:
        with redirect_stdout(f):
            model.summary()


    history = LossHistory()

    num_train_examples = int(np.ceil(len(images) * (1 - validation_split)))
    num_valid_examples = len(images) - num_train_examples

    train_Gen = train_ex_Gen(batch_size, images[:num_train_examples], labels[:num_train_examples])
    valid_Gen = valid_ex_Gen(batch_size, images[num_train_examples:], labels[num_train_examples:])

   # images, labels = shuffle(images, labels, random_state=0)
    print('beginning training')

    model.fit_generator(train_Gen,
                        steps_per_epoch=np.ceil(num_train_examples / batch_size),
                        validation_data=valid_Gen,
                        validation_steps=np.ceil(num_valid_examples / batch_size),
                        callbacks=[history],
                        epochs=num_epochs)


    model.save(write_dir + 'CNN_from_scratch.h5')
    print('Model saved.')
    np.savetxt(write_dir + 'history_loss.txt', np.array(history.loss))
    np.savetxt(write_dir + 'history_acc.txt', np.array(history.acc))
    np.savetxt(write_dir + 'history_batch_loss.txt', np.array(history.batch_loss))
    np.savetxt(write_dir + 'history_batch_accs.txt', np.array(history.batch_acc))
    np.savetxt(write_dir + 'history_val_acc.txt', np.array(history.val_acc))


if __name__ == "__main__":
   main()



