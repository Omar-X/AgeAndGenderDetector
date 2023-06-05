import pandas as pd
import numpy as np
import os

from PIL import Image
from keras_preprocessing.image import load_img
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

BASE_DIR = 'UTKFace'

# Load the data
# labels - age, gender, ethnicity
gender_dict = {0: 'Male', 1: 'Female'}
image_paths = []
age_labels = []
gender_labels = []
number_of_images = len(os.listdir(BASE_DIR))
category_age = np.zeros((number_of_images, 7), dtype='float32')


def age_group(Person_age):  # Function to group the ages into 7 categories
    if Person_age >= 60:
        return 6
    else:
        return int(Person_age / 10)


def images_extractor(images):  # Function to read and resize the images
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape((len(features), 128, 128, 1))
    return features


def get_age_from_group(age_list):
    largest_value = max(age_list)
    for index, age in enumerate(age_list):
        if age == largest_value:
            if index != 6:
                return "{}-{}".format(index * 10, index * 10 + 9)
            else:
                return "60+"


def predictor(image_index):  # get the prediction of the model and compare it with the original image
    print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", age_labels[image_index])
    predict = model.predict(Modified_Input[image_index].reshape(1, 128, 128, 1))
    predict_gender = gender_dict[round(predict[0][0][0])]
    predict_age = get_age_from_group(predict[1][0])
    print("Predicted Gender:", predict_gender, "Predicted Age:", predict_age)
    plt.axis('off')
    plt.imshow(Modified_Input[image_index].reshape(128, 128), cmap='gray')
    plt.show()


# Read the images and store them in a dataframe + create the labels for ages and genders
for index, filename in tqdm(enumerate(os.listdir(BASE_DIR))):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    category_age[index] = to_categorical(age_group(age), num_classes=7)
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

df = pd.DataFrame()  # create a dataframe
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# Preprocess the data
Modified_Input = images_extractor(df['image']) / 255.0

y_gender = np.array(df['gender'])
y_age = np.array(category_age)

input_shape = (128, 128, 1)

inputs = Input(input_shape)
# convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
drop_1 = Dropout(0.3)(maxp_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(drop_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.2)(dense_1)
dropout_2 = Dropout(0.2)(dense_2)

hidden_dense_1 = Dense(512, activation='relu')(dropout_1)
hidden_dense_2 = Dense(512, activation='relu')(dropout_2)

hidden_dropout_1 = Dropout(0.3)(hidden_dense_1)
hidden_dropout_2 = Dropout(0.3)(hidden_dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(hidden_dropout_1)
output_2 = Dense(7, activation='softmax', name='age_out')(hidden_dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

history = model.fit(x=Modified_Input, y=[y_gender, y_age], batch_size=32, epochs=10, validation_split=0.2)

# save the model
destFile = './AgeGenderDetector.h5'
model.save(destFile)

predictor(5)

