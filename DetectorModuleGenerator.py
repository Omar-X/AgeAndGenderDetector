from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from keras.layers import MaxPooling2D
from tqdm import tqdm
import random


def age_group(age):
    if 0 <= age < 10:
        return 0
    elif age < 20:
        return 1
    elif age < 30:
        return 2
    elif age < 40:
        return 3
    elif age < 50:
        return 4
    elif age < 60:
        return 5
    else:
        return 6


path = "/kaggle/input/utkface-new/UTKFace/"
files = os.listdir(path)
size = len(files)
print("Total samples:", size)

import cv2

images = []
ages = []
genders = []
for file in tqdm(files):
    image = cv2.imread(path + file, 0)
    image = cv2.resize(image, dsize=(200, 200))
    image = image.reshape((image.shape[0], image.shape[1], 1))
    images.append(image)
    split_var = file.split('_')
    ages.append(split_var[0])
    genders.append(int(split_var[1]))

m = len(files)
target = np.zeros((m, 7), dtype='float32')
age = np.zeros((m, 1), dtype='float32')
category_age = np.zeros((m, 7), dtype='float32')

features = np.zeros((m, 200, 200, 1), dtype='float32')
randomnums = random.sample(range(0, m), m)
for i in range(m):
    age[i] = age_group(int(ages[randomnums[i]]))
    features[i] = images[randomnums[i]]
category_age = to_categorical(age, 7)
target[:, :] = category_age
features = features / 255

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=7, padding="same", activation="relu", input_shape=(200, 200, 1)))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(250, activation="relu"))
model.add(Dense(7, activation="softmax"))

model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])
model.summary()

test_set = [x_test, y_test]
csv_logger = tf.keras.callbacks.CSVLogger('agedetector.csv')

model.fit(x_train, y_train, validation_data=test_set, epochs=5, batch_size=16, callbacks=[csv_logger])

dest = './agedetector.h5'
model.save(dest)
