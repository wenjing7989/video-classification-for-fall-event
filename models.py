import sys

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D, MaxPooling1D, AveragePooling1D)
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization

class mymodels():
    def __init__(self, nb_classes, modelname, num_cuts, img_size=[240,320,3], saved_model=None):
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.num_cuts = num_cuts

        metrics = ['accuracy']

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
            print(self.model.summary())
        elif modelname == 'crnn':
            print("Loading RNN model.")
            self.input_shape = [None]+img_size
            self.model = self.crnn()
        elif modelname == 'conv2':
            print("Loading Conv2D")
            self.input_shape = img_size
            self.model = self.conv2()
        elif modelname == 'conv3':
            print("Loading Conv3D")
            self.input_shape = [num_cuts]+img_size
            self.model = self.conv3()
        elif modelname == 'mcnn':
            print("Loading multi-frame CNN model.")
            self.input_shape = [num_cuts]+img_size
            self.model = self.mcnn()
        else:
            print("Unknown network!")
            sys.exit()

        #optimizer = optimizers.Adam(lr=1e-4)
        optimizer = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
            metrics=metrics)

    def conv3(self):
        model = Sequential()
        model.add(Conv3D(16, (3, 3, 3), padding='same', input_shape=self.input_shape,
            activation='relu'))
        model.add(MaxPooling3D((1,2,2)))
        model.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2,2,2)))
        model.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2,2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        print(model.summary())

        return model

    def crnn_small(self):
        model = Sequential()
        model.add(TimeDistributed(Conv2D(16,(3,3), padding='same', activation='relu'),
            input_shape=self.input_shape))
        #model.add(TimeDistributed((BatchNormalization())))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Conv2D(16,(3,3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Conv2D(16,(3,3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Conv2D(16,(3,3), padding='same', activation='relu')))
        #model.add(TimeDistributed(MaxPooling2D()))
        #model.add(TimeDistributed(Conv2D(16,(3,3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256))
        #model.add(LSTM(128, return_sequences=True))
        #model.add(Flatten())
        # model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(self.nb_classes, activation='softmax'))
        print(model.summary())

        return model


    def conv2(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=self.input_shape,
            activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        print(model.summary())

        return model

    def conv2_pre(self):
        input_tensor = Input(shape=self.input_shape)
        base_model = VGG16(input_tensor=input_tensor, weights='imagenet',
            include_top=False)
        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
        for layer in base_model.layers:
            layer.trainable = True
        print(model.summary())

        return model

    def crnn(self):
        base_model = VGG16(weights='imagenet', include_top=False)
        input_layer = Input(shape=self.input_shape)
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(256)(x)
        # x = Flatten()(x)
        # x = Dropout(0.5)(x)
        # x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=x)
        for layer in base_model.layers[-3:]:
            layer.trainable = True
        print(model.summary())

        return model

    def mcnn(self):
        input_layer = Input(shape=self.input_shape)
        x = TimeDistributed(Flatten())(input_layer)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(512, activation='relu'))(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.nb_classes, activation='softmax'))(x)
        x = MaxPooling1D(pool_size=self.input_shape[0])(x)
        x = Flatten()(x)
        model = Model(inputs=input_layer, outputs=x)
        print(model.summary())

        return model

    def mcnn3_pre(self):
        base_model = VGG16(weights='imagenet', include_top=False)
        input_layer = Input(shape=self.input_shape)
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dropout(0.5))(x)
        x = TimeDistributed(Dense(512, activation='relu'))(x)
        x = TimeDistributed(Dropout(0.5))(x)
        x = TimeDistributed(Dense(self.nb_classes))(x)
        x = MaxPooling1D(pool_size=self.input_shape[0])(x)
        x = Flatten()(x)
        x = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=x)
        for layer in base_model.layers:
            layer.trainable = False
        print(model.summary())

        return model

    def mcnn2_pre(self):
        base_model = VGG16(weights='imagenet', include_top=False)
        input_layer = Input(shape=self.input_shape)
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dropout(0.5))(x)
        x = TimeDistributed(Dense(512, activation='relu'))(x)
        x = MaxPooling1D(pool_size=self.input_shape[0])(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=x)
        for layer in base_model.layers:
            layer.trainable = False
        print(model.summary())

        return model

    def mcnn1_pre(self):
        base_model = VGG16(weights='imagenet', include_top=False)
        input_layer = Input(shape=self.input_shape)
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(Flatten())(x)
        x = MaxPooling1D(pool_size=self.input_shape[0])(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=x)
        for layer in base_model.layers:
            layer.trainable = False
        print(model.summary())

        return model


if __name__=='__main__':
    a=mymodels(5, 'crnn', 5)
