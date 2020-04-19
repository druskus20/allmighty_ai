from tensorflow.keras import Sequential, optimizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, TimeDistributed, Flatten, GRU, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer


def get_model(selected_model='CNN+RNN'):
    if selected_model == 'CNN+RNN':
        model = Sequential()

        model.add(InputLayer(input_shape=(5, 270, 480, 3)))

        model.add(TimeDistributed(Convolution2D(32, (4, 4), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(Convolution2D(32, (4, 4), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
        model.add(TimeDistributed(Dropout(0.25)))
        print(model.output_shape)

        model.add(TimeDistributed(Convolution2D(16, (3, 3), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
        model.add(TimeDistributed(Dropout(0.25)))
        print(model.output_shape)

        model.add(TimeDistributed(Flatten()))
        print(model.output_shape)

        model.add(GRU(256, kernel_initializer=initializers.RandomNormal(stddev=0.001)))  # 128
        model.add(Dropout(0.25))
        print(model.output_shape)

        model.add(Dense(100))
        print(model.output_shape)

        model.add(Dense(80))
        print(model.output_shape)

        model.add(Dense(40))
        print(model.output_shape)

        model.add(Dense(9, activation='sigmoid'))
        print(model.output_shape)

        opt = optimizers.rmsprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    elif selected_model == 'CNN+MLP':
        model = Sequential()

        model.add(InputLayer(input_shape=(5, 270, 480, 3)))

        model.add(TimeDistributed(Convolution2D(16, (4, 8), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(Convolution2D(16, (4, 4), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
        model.add(TimeDistributed(Dropout(0.25)))
        print(model.output_shape)

        model.add(TimeDistributed(Convolution2D(12, (3, 3), data_format='channels_last')))
        model.add(TimeDistributed(Activation('relu')))
        print(model.output_shape)

        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5), data_format='channels_last')))
        model.add(TimeDistributed(Dropout(0.25)))
        print(model.output_shape)

        model.add(Flatten())
        print(model.output_shape)

        model.add(Dense(300))
        print(model.output_shape)
        model.add(Dense(100))
        print(model.output_shape)
        print(model.output_shape)
        model.add(Dense(9, activation='sigmoid'))
        print(model.output_shape)

        opt = optimizers.rmsprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
