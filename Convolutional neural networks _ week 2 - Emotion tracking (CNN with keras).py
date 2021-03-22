from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')

#################################
#  Building the model in Keras  #
#################################

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    Arguments:
            input_shape -- shape of the images of the dataset (height, width, channels) as a tuple.
                Note that this does not include the 'batch' as a dimension.
    Returns:
            model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2))(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)

    # DENSE -> BN -> relu -> Dropout -> DENSE
    X = Dense(128)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

###################################################
#  Create, compile, train and evaluate the model  #
###################################################

happyModel = HappyModel(X_train.shape[1:])

happyModel.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

happyModel.fit(X_train, Y_train, epochs=5, batch_size=16, validation_data=(X_test, Y_test))

happyModel.evaluate(x=X_test, y=Y_test)
