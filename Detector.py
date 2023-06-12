import os
import sys

import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from threading import Thread

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


class FaceFounder:
    def __init__(self, cameraIndex=0, width=640, height=480):
        self.capture = None
        self.cameraIndex = cameraIndex
        self.width = width
        self.height = height
        self.faceDetector = None
        self.text = None
        self.image = None
        self.facebox = None
        self.out_of_frame = False
        self.camera_is_closed = False

    def start(self):
        self.faceDetector = FaceDetector()
        self.capture = cv2.VideoCapture(self.cameraIndex)
        self.capture.set(3, self.width)
        self.capture.set(4, self.height)

    def _grepImage(self):
        success, img = self.capture.read()
        self.image = cv2.flip(img, 1)
        # grey image
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def _findFace(self):
        self.image, bbox = self.faceDetector.findFaces(self.image)
        if bbox:
            self.out_of_frame = False
            self.facebox = list(bbox[0]["bbox"])
            if self.facebox[0] < 0:
                self.out_of_frame = True
                self.facebox[0] = 0
            if self.facebox[1] < 0:
                self.out_of_frame = True
                self.facebox[1] = 0

            if self.facebox[0] + self.facebox[2] > self.width:
                self.out_of_frame = True
                self.facebox[2] = self.width - self.facebox[0]
            if self.facebox[1] + self.facebox[3] > self.height:
                self.out_of_frame = True
                self.facebox[3] = self.height - self.facebox[1]

            self.faceImage = self.image[self.facebox[1]:self.facebox[1] + self.facebox[3],
                             self.facebox[0]:self.facebox[0] + self.facebox[2]]
            return True
        return False

    def run(self, inThread=False):
        if inThread:
            Thread(target=self._run, args=()).start()
        else:
            self._run()

    def _run(self):
        while True:
            self._grepImage()
            if not self._findFace():
                self.text = "Face Not Found"
            cv2.putText(self.image, self.text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("Image", self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.camera_is_closed = True
                break
        self.capture.release()
        cv2.destroyAllWindows()


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

    def predict(self, printout=True):
        if self.model and self.image_array is not None:
            self.prediction = self.model.predict(self.image_array.reshape(1, 128, 128, 1))
            self.predict_gender = gender_dict[round(self.prediction[0][0][0])]
            self.predict_age = get_age_from_group(self.prediction[1][0])
            if printout:
                print("Predicted Gender:", self.predict_gender, "Predicted Age:", self.predict_age)
        else:
            if printout:
                print("Model or image not found")

    def showResults(self):
        if self.prediction:
            plt.axis('off')
            plt.imshow(self.image_array.reshape(128, 128), cmap='gray')
            plt.title(("Predicted Gender:", self.predict_gender, "Predicted Age:", self.predict_age))
            plt.show()
        else:
            print("Prediction not found")


class Project:
    def __init__(self, modelPath):
        self.faceFounder = FaceFounder()
        self.ageGenderDetector = AgeGenderDetector(modelPath)

    def start(self):
        self.faceFounder.start()
        self.ageGenderDetector.loadModel()

    def run(self):
        self.faceFounder.run(inThread=True)
        while True:
            if self.faceFounder.facebox and not self.faceFounder.camera_is_closed:
                image_array = np.array(self.faceFounder.faceImage)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                image_array = np.array(cv2.resize(image_array, (128, 128))) / 255.0
                self.ageGenderDetector.image_array = image_array

                self.ageGenderDetector.predict(printout=False)
                text = self.ageGenderDetector.predict_gender + " " + self.ageGenderDetector.predict_age
                self.faceFounder.text = text


if __name__ == "__main__":
    clear()
    model_Path = "../AgeAndGenderDetector/AgeGenderDetector.h5"  # Path to model, change it.
    project = Project(model_Path)
    project.start()
    project.run()
