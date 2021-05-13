from enum import Enum, auto, IntEnum

import tensorflow as tf  # TF2
from PIL import Image
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PoseEstim:

    def __init__(self, model):
        self.interpreter = tf.lite.Interpreter(
            model_path=model,
            num_threads=4)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.img_height = self.input_details[0]['shape'][1]
        self.img_width = self.input_details[0]['shape'][2]

    class BodyPart(IntEnum):

        NOSE = 0
        LEFT_EYE = auto()
        RIGHT_EYE = auto()
        LEFT_EAR = auto()
        RIGHT_EAR = auto()
        LEFT_SHOULDER = auto()
        RIGHT_SHOULDER = auto()
        LEFT_ELBOW = auto()
        RIGHT_ELBOW = auto()
        LEFT_WRIST = auto()
        RIGHT_WRIST = auto()
        LEFT_HIP = auto()
        RIGHT_HIP = auto()
        LEFT_KNEE = auto()
        RIGHT_KNEE = auto()
        LEFT_ANKLE = auto()
        RIGHT_ANKLE = auto()

    class Position(object):
        def __init__(self):
            self.x = 0
            self.y = 0

    class KeyPoint(object):
        def __init__(self):
            self.bodyPart: PoseEstim.BodyPart = PoseEstim.BodyPart.NOSE
            self.position: PoseEstim.Position = PoseEstim.Position()
            self.score = 0.0

    class Person(object):
        def __init__(self):
            self.keyPoints = list()
            self.score = 0.0

    def fwd(self, img):

        input_data = np.expand_dims(img, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        heatmaps = self.interpreter.get_tensor(self.output_details[0]['index'])
        offsets = self.interpreter.get_tensor(self.output_details[1]['index'])

        height = heatmaps.shape[1]
        width = heatmaps.shape[2]
        numKeypoints = heatmaps.shape[3]

        keypointPositions = [0 for i in range(numKeypoints)]

        for keypoint in range(numKeypoints):
            maxVal = heatmaps[0][0][0][keypoint]
            maxRow = 0
            maxCol = 0
            for row in range(height):
                for col in range(width):
                    if heatmaps[0][row][col][keypoint] > maxVal:
                        maxVal = heatmaps[0][row][col][keypoint]
                        maxRow = row
                        maxCol = col

            keypointPositions[keypoint] = (maxRow, maxCol)

        xCoords = [0 for i in range(numKeypoints)]
        yCoords = [0 for i in range(numKeypoints)]
        confidenceScores = [0 for i in range(numKeypoints)]

        for idx, position in enumerate(keypointPositions):
            positionY = keypointPositions[idx][0]
            positionX = keypointPositions[idx][1]
            yCoords[idx] = position[0] / (height - 1) * self.img_height + offsets[0][positionY][positionX][idx]
            xCoords[idx] = position[1] / (width - 1) * self.img_width + \
                           offsets[0][positionY][positionX][idx + numKeypoints]

            confidenceScores[idx] = sigmoid(heatmaps[0][positionY][positionX][idx])

        # result = np.squeeze(output_data)

        person = PoseEstim.Person()
        keypointList = [PoseEstim.KeyPoint() for i in range(numKeypoints)]

        totalScore = 0.0

        for it in PoseEstim.BodyPart:
            idx = it.value
            keypointList[idx].bodyPart = it
            keypointList[idx].position.x = xCoords[idx]
            keypointList[idx].position.y = yCoords[idx]
            keypointList[idx].score = confidenceScores[idx]
            totalScore += confidenceScores[idx]

        person.keyPoints = keypointList
        person.score = totalScore / numKeypoints

        return person
