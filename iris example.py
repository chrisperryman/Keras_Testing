import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Set random seed
np.random.seed(2018)


# Load in iris dataset
iris = np.load("data/iris.npy")

# Shuffle rows
np.random.shuffle(iris)

iris_labels = iris[:, 4]


iris_onehot = to_categorical(iris_labels)
iris_onehot

# Initiate Sequential Model
model = Sequential()

model.add(Dense(15, activation="sigmoid", input_shape=(4,)))
model.add(Dense(3, activation="softmax"))

model.summary()

# Compile Model
model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model.fit(iris[:, :4],
          iris_onehot, epochs=500,
          batch_size=20,
          validation_split=0.2)
