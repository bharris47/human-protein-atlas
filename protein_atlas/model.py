from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPooling2D, Dense, Dropout
from keras.models import Model


def _conv_block(x, filters, kernel_size, activation, pool=None):
    hidden = Conv2D(filters, kernel_size, activation=activation)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Conv2D(filters, kernel_size, activation=activation)(hidden)
    hidden = BatchNormalization()(hidden)
    if pool:
        return MaxPool2D(pool_size=pool)(hidden)
    return hidden


def simple_convnet(input_shape, n_classes, base_filters, activation, fc_size, dropout, classifier_activation):
    image = Input(shape=input_shape)

    conv_1 = Conv2D(base_filters, 7, activation=activation)(image)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = _conv_block(conv_1, filters=base_filters, kernel_size=3, activation=activation, pool=2)
    conv_3 = _conv_block(conv_2, base_filters * 2, kernel_size=3, activation=activation)
    conv_4 = _conv_block(conv_3, base_filters * 2, kernel_size=3, activation=activation, pool=2)
    conv_5 = _conv_block(conv_4, base_filters * 4, kernel_size=3, activation=activation)
    conv_6 = _conv_block(conv_5, base_filters * 4, kernel_size=3, activation=activation, pool=2)
    conv_7 = _conv_block(conv_6, base_filters * 2, kernel_size=1, activation=activation)

    hidden = GlobalMaxPooling2D()(conv_7)

    fc_1 = Dense(fc_size, activation=activation)(hidden)
    fc_1 = Dropout(dropout)(fc_1)
    fc_2 = Dense(fc_size, activation=activation)(fc_1)
    fc_2 = Dropout(dropout)(fc_2)

    predictions = Dense(n_classes, activation=classifier_activation)(fc_2)
    return Model(image, predictions)
