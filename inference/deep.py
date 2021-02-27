import tensorflow_hub as hub
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf


def create_model():
    gan_layer = hub.KerasLayer('https://tfhub.dev/google/progan-128/1', trainable=False)
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(gan_layer)
    return model

# model = create_model()