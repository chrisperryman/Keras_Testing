import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Set random seed
np.random.seed(2018)

walking = np.load("data/walking.npy")
walking.shape

import matplotlib.pyplot as plt

def plot_series(series):
    # x-channel
    plt.plot(series[:, 0], color="red")
    # y-channel
    plt.plot(series[:, 1], color="green")
    # z-channel
    plt.plot(series[:, 2], color="blue")

plot_series(walking[408, :10, :])
plot_series(walking[100, :, :])

# Prepare data for ML model
walking_labels = np.load("data/walking_labels.npy")
set(walking_labels)

m = walking.shape[0]
indices = [x for x in range(m)]
np.random.shuffle(indices)

train_indices = indices[:int(m*0.6)]
val_indices = indices[int(m*0.6):int(m*0.8)]
test_indices = indices[int(m*0.8):]

X_train = walking[train_indices, :, :]
X_val = walking[val_indices, :, :]
X_test = walking[test_indices, :, :]

y_train = to_categorical(walking_labels[train_indices])
y_val = to_categorical(walking_labels[val_indices])
y_test = to_categorical(walking_labels[test_indices])

# Shape model

model = Sequential()

from keras.layers import Conv1D, MaxPooling1D, Flatten

model.add(Conv1D(filters=30, kernel_size=40, strides=2, activation="relu", input_shape=(260, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=30, kernel_size=10, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.output_shape

model.add(Flatten())
model.output_shape


model.add(Dense(100, activation="sigmoid"))
model.add(Dense(15, activation="softmax"))


model.summary()

# Compile Model

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_val, y_val))

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict_classes(X_test)

print(classification_report(np.argmax(y_test, axis=1), y_pred))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

