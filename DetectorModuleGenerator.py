"""
The Purpose of this program is to generate a model that can detect the age as well as the gender of a person
from a given image. The model is trained on the UTKFace dataset. The model is saved as a .h5 file.
"""
import numpy as np
import os
from tqdm import tqdm
import random
import cv2

from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D  # Importing the required libraries
from keras.utils import to_categorical  # for the model
from keras.models import Sequential  # to be trained
from sklearn.model_selection import train_test_split  # on the dataset
import tensorflow as tf


def age_group(age):  # Function to group the ages into 7 categories
    if age >= 60:
        return 6
    else:
        return int(age / 10)


path = "UTKFace/"  # Path to the dataset
filenames = os.listdir(path)
size = len(filenames)
print("Total samples:", size)

images = []  # list to store the images
ages = []  # list to store the ages
genders = []  # list to store genders
for file in tqdm(filenames):
    image = cv2.imread(path + file, 0)
    image = cv2.resize(image, dsize=(200, 200))
    image = image.reshape((image.shape[0], image.shape[1], 1))
    images.append(image)
    split_var = file.split('_')
    ages.append(split_var[0])
    genders.append(int(split_var[1]))

AgeTarget = np.zeros((size, 1), dtype='float32')
GenderTarget = np.zeros((size, 1), dtype='float32')
age = np.zeros((size, 1), dtype='float32')
# category_age = np.zeros((size, 7), dtype='float32')

features = np.zeros((size, 200, 200, 1), dtype='float32')
randomnums = random.sample(range(0, size), size)
for i in range(size):
    age[i] = age_group(int(ages[randomnums[i]]))
    GenderTarget[i] = genders[randomnums[i]]
    features[i] = images[randomnums[i]]
# add an extra dimension to the array to make it suitable for the model
AgeTarget = age
features = features / 255.0


# split the dataset into training and testing sets
x_train, x_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(features, AgeTarget,
                                                                                           GenderTarget, test_size=0.2,
                                                                                           random_state=42)

# first and second convolutional layers
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=7, padding="same", activation="relu", input_shape=(200, 200, 1)))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

# third and fourth convolutional layers
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))

# fifth convolutional layer
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

# flattening
model.add(Flatten())

# fully connected layers
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(250, activation="relu"))

# output layers
model.add(Dense(1, activation="softmax", name="age_output"))
model.add(Dense(1, activation="softmax", name="gender_output"))

model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])
model.summary()


# train the model
test_set = [x_test, y_age_test, y_gender_test]
csv_logger = tf.keras.callbacks.CSVLogger('agedetector.csv')

# train the model
model.fit(x_train, [y_age_train, y_gender_train], validation_data=test_set, epochs=10, batch_size=32, callbacks=[csv_logger])

# save the model
destFile = './AgeGenderDetector.h5'
model.save(destFile)
