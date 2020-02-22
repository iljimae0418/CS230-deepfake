import torch
import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mplimg
from sklearn.utils import shuffle

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
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

    batch_size = 32
    num_epochs = 2
    validation_split = 0.15
    n_hidden_units = [1096]

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

    base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten())
    for n in n_hidden_units:
        top_model.add(Dense(n, activation="relu"))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n, activation="relu"))
        top_model.add(Dense(1, activation='sigmoid'))

        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        history = LossHistory()

        num_train_examples = int(np.ceil(len(images) * (1 - validation_split)))
        num_valid_examples = len(images) - num_train_examples

        train_Gen = train_ex_Gen(batch_size, images[:num_train_examples], labels[:num_train_examples])
        valid_Gen = valid_ex_Gen(batch_size, images[num_train_examples:], labels[num_train_examples:])

       # images, labels = shuffle(images, labels, random_state=0)
        print('beginning training with ' + str(n) + ' hidden units')

        model.fit_generator(train_Gen,
                            steps_per_epoch=np.ceil(num_train_examples / batch_size),
                            validation_data=valid_Gen,
                            validation_steps=np.ceil(num_valid_examples / batch_size),
                            callbacks=[history],
                            epochs=num_epochs)

        '''
        model.fit(x=np.array(images),
                  y=np.array(labels),
                  batch_size=batch_size,
                  validation_split=validation_split,
                  callbacks=[history],
                  epochs=num_epochs,
                  shuffle=True)
        '''

        if not (os.path.exists(output_dir + '/finetuning/')):
            os.makedirs(output_dir + '/finetuning/')

        model.save(output_dir + '/finetuning/vgg16_finetuned_' + str(n) + '_h_units_4_trainable_layers.h5')
        print('Model saved.')
        np.savetxt(output_dir + '/finetuning/history_loss_' + str(n) + '_h_units_4_trainable_layers.txt', np.array(history.loss))
        np.savetxt(output_dir + '/finetuning/history_acc_' + str(n) + '_h_units_4_trainable_layers.txt', np.array(history.acc))
        np.savetxt(output_dir + '/finetuning/history_batch_loss_' + str(n) + '_h_units_4_trainable_layers.txt', np.array(history.batch_loss))
        np.savetxt(output_dir + '/finetuning/history_batch_acc_' + str(n) + '_h_units_4_trainable_layers.txt', np.array(history.batch_acc))
        np.savetxt(output_dir + '/finetuning/history_val_acc_' + str(n) + '_h_units_4_trainable_layers.txt', np.array(history.val_acc))


if __name__ == "__main__":
   main()



