# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image
from PoseEstim import PoseEstim
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
from enum import Enum, auto, IntEnum

import tensorflow as tf  # TF2
from PIL import Image
import numpy as np
import cv2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
        self.bodyPart : BodyPart = BodyPart.NOSE
        self.position : Position = Position()
        self.score = 0.0


class Person(object):
    def __init__(self):
        self.keyPoints = list()
        self.score = 0.0


def main():
    interpreter = tf.lite.Interpreter(
        model_path="posenet_model.tflite",
        num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    # print(input_details)
    # print(output_details)

    img_height = input_details[0]['shape'][1]
    img_width = input_details[0]['shape'][2]
    img = Image.open('./tmp/grace_hopper.bmp').resize((img_width, img_height))
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    heatmaps = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])

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
        yCoords[idx] = position[0] / (height - 1) * img_height + offsets[0][positionY][positionX][idx]
        xCoords[idx] = position[1] / (width - 1) * img_width + \
                       offsets[0][positionY][positionX][idx + numKeypoints]

        confidenceScores[idx] = sigmoid(heatmaps[0][positionY][positionX][idx])

    # result = np.squeeze(output_data)

    person = Person()
    keypointList = [KeyPoint() for i in range(numKeypoints)]


    totalScore = 0.0

    for it in BodyPart:
        idx = it.value
        keypointList[idx].bodyPart = it
        keypointList[idx].position.x = xCoords[idx]
        keypointList[idx].position.y = yCoords[idx]
        keypointList[idx].score = confidenceScores[idx]
        totalScore += confidenceScores[idx]


    person.keyPoints = keypointList
    person.score = totalScore / numKeypoints

    #return person


    #img.show()

    result = np.array(img)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    for kp in person.keyPoints:
        result = cv2.line(result, (int(kp.position.x), int(kp.position.y)), (int(kp.position.x), int(kp.position.y)), (0, 0, 255), 5)

    cv2.imshow('result', result)
    cv2.waitKey(0)

    print("dx")


"""
def keyPointsToNormnparray(img_width, img_height, keyPoints):
    return list(map(lambda x: np.array([x.position.x / img_width, x.position.y / img_height]), keyPoints))

def main():
    net = PoseEstim("posenet_model.tflite")
    cap = cv2.VideoCapture("stable.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)


    # totData = list()

    countFlag = False
    count = 0
    standState = None
    diff_list = None
    while cap.isOpened():  # cap 정상동작 확인
        ret, frame = cap.read()
        # 프레임이 올바르게 읽히면 ret은 True
        if not ret:
            print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
            break

        img = cv2.resize(frame, dsize=(net.img_width, net.img_height), interpolation=cv2.INTER_AREA)
        person = net.fwd(img)

        if countFlag:
            currentState = np.concatenate((keyPointsToNormnparray(net.img_width, net.img_height, person.keyPoints[0:5]),
                                         keyPointsToNormnparray(net.img_width, net.img_height,
                                                                person.keyPoints[11:15])))
            diff = standState - currentState
            diff_list = np.append(diff_list, [diff], axis=0)

            for i in range(9):
                plt.subplot(331+i)
                plt.plot(diff_list[:, i, 1])
            plt.show()

            # print(diff)


        """
        tempData = list()
        for kp in person.keyPoints:
            tempData.append([kp.bodyPart.value, kp.position.x, kp.position.y])
        totData.append(tempData)
        """
        # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        for kp in person.keyPoints:
            img = cv2.line(img, (int(kp.position.x), int(kp.position.y)),
                              (int(kp.position.x), int(kp.position.y)), (0, 0, 255), 5)

        cv2.imshow('Otter', img)
        input_key = cv2.waitKey(1)
        if input_key == ord('q'):
            break
        elif input_key == ord('d'):
            for kp in person.keyPoints:
                norm_x = kp.position.x / net.img_width
                norm_y = kp.position.y / net.img_height
                print(kp.bodyPart.name, " : ", norm_x, ", ", norm_y)
            cv2.waitKey(0)
        elif input_key == ord('s'):
            countFlag = True
            standState = np.concatenate((keyPointsToNormnparray(net.img_width, net.img_height, person.keyPoints[0:5]), keyPointsToNormnparray(net.img_width, net.img_height, person.keyPoints[11:15])))
            diff_list = np.empty((1, 9, 2))


            # 작업 완료 후 해제
    cap.release()
    # totData = np.array(totData)
    # np.save("heebin.npy", totData);

    """
    img = Image.open('./tmp/grace_hopper.bmp').resize((net.img_width, net.img_height))
    person = net.fwd(img)

    result = np.array(img)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    for kp in person.keyPoints:
        result = cv2.line(result, (int(kp.position.x), int(kp.position.y)),
                          (int(kp.position.x), int(kp.position.y)), (0, 0, 255), 5)


    cv2.imshow('result', result)
    cv2.waitKey(0)
    """


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
