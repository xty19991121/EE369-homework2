import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D

from keras.optimizers import RMSprop

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10

# download and load the data (split them between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# expand the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# define a simple CNN model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=128, verbose=2)

# evaluate
score_train = model.evaluate(x_train, y_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100, score_train[1]*100))
score_test = model.evaluate(x_test, y_test, verbose=0)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0]*100, score_test[1]*100))

