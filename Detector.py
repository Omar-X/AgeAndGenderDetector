import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

gender_dict = {0: 'Male', 1: 'Female'}


def get_age_from_group(age_list):
    largest_value = max(age_list)
    for index, age in enumerate(age_list):
        if age == largest_value:
            if index != 6:
                return "{}-{}".format(index * 10, index * 10 + 9)
            else:
                return "60+"


def clear():
    if sys.platform == "linux":
        os.system("clear")
    else:
        os.system("cls")


class AgeGenderDetector:

    def __init__(self, modelPath):
        self.predict_age = None
        self.predict_gender = None
        self.prediction = None
        self.image_array = None
        self.figurePath = None
        self.model = None
        self.modelPath = modelPath

    def loadModel(self):
        if os.path.exists(self.modelPath):
            self.model = tf.keras.models.load_model(self.modelPath)
            print("Model loaded")
        else:
            print("Model not found")

    def loadFigure(self, figurePath):
        self.figurePath = figurePath
        if os.path.exists(self.figurePath):
            image_bytes = cv2.imread(self.figurePath, 0)
            self.image_array = np.array(cv2.resize(image_bytes, (128, 128))) / 255.0
            print("Image loaded")
        else:
            print("Image not found")

    def predict(self):
        if self.model and self.image_array is not None:
            self.prediction = self.model.predict(self.image_array.reshape(1, 128, 128, 1))
            self.predict_gender = gender_dict[round(self.prediction[0][0][0])]
            self.predict_age = get_age_from_group(self.prediction[1][0])
            print("Predicted Gender:", self.predict_gender, "Predicted Age:", self.predict_age)
        else:
            print("Model or image not found")

    def showResults(self):
        if self.prediction:
            plt.axis('off')
            plt.imshow(self.image_array.reshape(128, 128), cmap='gray')
            plt.title(("Predicted Gender:", self.predict_gender, "Predicted Age:", self.predict_age))
            plt.show()
        else:
            print("Prediction not found")
