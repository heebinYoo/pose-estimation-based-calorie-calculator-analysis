# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from tqdm import tqdm


def plotwhole(data):
    for pose in tqdm(data):
        # print(pose)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(pose[:, 0], pose[:, 1], 'ro')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.invert_yaxis()
        plt.show()


def plotgraph(data, draw_feature=np.ones(shape=(17, 2), dtype=bool)):
    fig = plt.figure(figsize=[20, 100])
    for i in range(1, 34, 2):
        if draw_feature[int(i / 2), 0]:
            ax = fig.add_subplot(17, 2, i)
            ax.set_ylim(-1.1, 1.1)
            ax.plot(data[:, int(i / 2), 0])

        if draw_feature[int(i / 2), 1]:
            ax = fig.add_subplot(17, 2, i + 1)
            ax.set_ylim(-1.1, 1.1)
            ax.plot(data[:, int(i / 2), 1])
    plt.show()


def plotcorrgraph(data):
    fig = plt.figure(figsize=[20, 100])
    for i in range(1, 34, 2):
        ax = fig.add_subplot(17, 2, i)
        ax.set_ylim(-0.1, 30)
        ax.plot(data[:, int(i / 2), 0])

        ax = fig.add_subplot(17, 2, i + 1)
        ax.set_ylim(-0.1, 30)
        ax.plot(data[:, int(i / 2), 1])
    fig.show()
    return fig


raw = np.load("heebin.npy")
raw = np.delete(raw, 0, 2)
raw = np.clip(raw / 257, 0, 1)

data = raw[90:860].copy()
fake = np.concatenate((raw[0:90].copy(), raw[860:-1].copy()))

# plotgraph(data)

# data_mean = np.apply_along_axis(lambda a: np.mean(a), 0, data)
stable = data[0].copy()
for i in range(len(data)):
    data[i] = data[i] - stable

data = gaussian_filter1d(data, sigma=3, axis=0)

# plotgraph(data)
corrlist = list()

for i in range(1, 34, 2):
    x_data = np.correlate(data[:, int(i / 2), 0], data[:, int(i / 2), 0], "full")
    y_data = np.correlate(data[:, int(i / 2), 1], data[:, int(i / 2), 1], "full")
    x_data = np.array([x_data]).T
    y_data = np.array([y_data]).T
    result = np.concatenate((x_data, y_data), axis=1)
    result = np.array([result])
    result = np.swapaxes(result, 0, 1)
    corrlist.append(result)

corrdata = np.concatenate(tuple(corrlist), axis=1)
corrdata = np.round(corrdata, 7)

# corrfig = plotcorrgraph(corrdata)


timings = [[], [], [], [], []]
good_feature = np.empty(shape=(17, 2), dtype=bool)
for point in range(17):
    for coord in range(2):
        maxima_x = argrelextrema(corrdata[:, point, coord], np.greater)[0]
        minima_x = argrelextrema(corrdata[:, point, coord], np.less)[0]

        if maxima_x.shape[0] == 9 and minima_x.shape[0] == 10:
            minima_x = minima_x[1:-1]

        elif maxima_x.shape[0] == 11 and minima_x.shape[0] == 10:
            maxima_x = maxima_x[1:-1]
            minima_x = minima_x[1:-1]

        # ax = corrfig.get_axes()[point * 2 + coord]
        # ax.plot(maxima_x, corrdata[maxima_x, point, coord], 'ro')
        # ax.plot(minima_x, corrdata[minima_x, point, coord], 'bo')
        # ax.text(200, 25, F'maxima : {maxima_x.shape[0]} minima : {minima_x.shape[0]}')

        if maxima_x.shape[0] == 9 and minima_x.shape[0] == 8:
            good_feature[point, coord] = True
            last_time = 0
            for i in range(5):
                timings[i].append((minima_x[i] + maxima_x[i]) / 2 - last_time)
                last_time = (minima_x[i] + maxima_x[i]) / 2

# check maxima, minima
# corrfig.show()
timings = np.array(timings)
timings = np.sort(timings, axis=1)

s_idx = 0.15 * (len(timings[0]) - 1)
s_idx = int(s_idx + 0.5)

e_idx = 0.85 * (len(timings[0]) - 1)
e_idx = int(e_idx + 0.5)
timings = np.mean(timings[:, s_idx:e_idx], axis=1)
timings = np.cumsum(timings).astype(int)

if timings[4] > len(data) - 1:
    timings[4] = len(data) - 1

# check cyclic point
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(raw[:, 0, 1])
ax.plot(timings+90, raw[timings+90, 0, 1], 'ro')
ax.plot(90, raw[90, 0, 1], 'ro')
fig.show()

sepreated_data = list()
start_p = 0
for i in range(5):
#     plotgraph(data[start_p:timings[i]], good_feature)
    sepreated_data.append(data[start_p:timings[i]])
    start_p = timings[i]


print(sepreated_data)



# for csv saving
sep_list = []
for i in range(5):
    sep_list.append(sepreated_data[i].reshape(-1, 34) + stable.reshape(34))

import dtw
res = dtw.dtw(sep_list[0], sep_list[1], keep_internals=True)

temp = (sep_list[0][res.index1] + sep_list[0][res.index2])/2
del sep_list[0:2]

for i in range(len(sep_list)):
    res = dtw.dtw(temp, sep_list[0], keep_internals=True)
    temp = (temp[res.index1] + sep_list[0][res.index2])/2
    del sep_list[0]



print(temp)







def makeTrueLabel(timings_data):
    result_list = [1, 2, 3, 4, 5]
    for t in timings_data:
        for itr_idx in range(-5, 6):
            if len(data) - 1 >= t + itr_idx:
                result_list.append(t + itr_idx)
    return np.array(result_list)



true_label = makeTrueLabel(timings)
mask = np.zeros(data.shape[0], dtype=bool)
mask[true_label] = True

stand_states = data[mask] + stable
stand_states = stand_states.reshape((stand_states.shape[0], -1))

nonstand_states = np.concatenate((data[~mask], fake))
nonstand_states = nonstand_states.reshape(nonstand_states.shape[0], -1)

X_data = np.concatenate((stand_states, nonstand_states))
Y_data = np.zeros(stand_states.shape[0] + nonstand_states.shape[0], dtype=bool)
Y_data[0:len(stand_states)] = True

from sklearn.metrics import confusion_matrix
from pprint import pprint
from sklearn.model_selection import train_test_split

"""
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    Y_data,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=1008)
"""


X_train, X_test, y_train, y_test = X_data, X_data, Y_data, Y_data
#TODO 잘 분해하는 방법을 알아야 하겠다.

# from sklearn import svm
# clf = svm.NuSVC(gamma='auto', nu=0.001, class_weight='balanced')
# clf.fit(X_train, y_train)
#
# pred = clf.predict(X_test)
# pprint(confusion_matrix(y_test, pred))

from sklearn.linear_model import LogisticRegressionCV

cls = LogisticRegressionCV(cv=10, scoring='precision', solver='liblinear', max_iter=500, class_weight='balanced',
                           n_jobs=-1)

cls.fit(X=X_train, y=y_train)
pred = cls.predict(X_test)
pprint(confusion_matrix(y_test, pred))

# from sklearn.ensemble import RandomForestClassifier
#
# random_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, class_weight='balanced', n_jobs=-1)
# random_clf.fit(X_train, y_train)
# pred = random_clf.predict(X_test)
# pprint(confusion_matrix(y_test, pred))
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, class_weight='balanced', n_jobs=-1)
# extra_clf.fit(X_train, y_train)
# pred = extra_clf.predict(X_test)
# pprint(confusion_matrix(y_test, pred))


print("done")

pred = cls.predict(raw.reshape(len(raw), -1))

print(90, timings+90)
pprint(np.argwhere(pred == True))


# import cv2
#
# video_file = "IMG_1208.MOV"  # 동영상 파일 경로
#
# cap = cv2.VideoCapture(video_file)  # 동영상 캡쳐 객체 생성  ---①
# if cap.isOpened():  # 캡쳐 객체 초기화 확인
#
#     for stand_frame in pred:
#         ret, img = cap.read()  # 다음 프레임 읽기      --- ②
#         if ret:  # 프레임 읽기 정상
#             if stand_frame:
#                 cv2.imshow(video_file, img)  # 화면에 표시  --- ③
#                 cv2.waitKey(1)  # 25ms 지연(40fps로 가정)   --- ④
#
#         else:  # 다음 프레임 읽을 수 없음,
#             break  # 재생 완료
# else:
#     print("can't open video.")  # 캡쳐 객체 초기화 실패
# cap.release()  # 캡쳐 자원 반납
# cv2.destroyAllWindows()


import dtw
# good_feature.reshape(34)

dist_mat = np.empty((len(sepreated_data), len(sepreated_data)))

for i, q1 in enumerate(sepreated_data):
    for j, q2 in enumerate(sepreated_data):
        query = q1.reshape(q1.shape[0], -1)
        query2 = q2.reshape(q2.shape[0], -1)
        dist_mat[i, j] = dtw.dtw(query, query2, keep_internals=True).normalizedDistance
# pprint(dist_mat)
# pprint(np.max(dist_mat))

template = query+stable.reshape(-1)

threshold = np.max(dist_mat)


random = raw[860:970].reshape(raw[860:970].shape[0], -1)
print(dtw.dtw(query, random, keep_internals=True).normalizedDistance)
# 3.151766807781363


routine_delay = 50
# 다음 운동 시작까지 이정도는 기다려야 하지 않을까?
# 한번 계산 쓰레드를 시작한 후에 routine_delay 프레임까지는 시작하지 마
last_task = -1

for i, stand_frame in enumerate(pred):
    if stand_frame and (last_task == -1 or last_task + routine_delay < i):
        for j in range(i+routine_delay, len(pred)):
            if pred[j]:
                query = raw[i:j].reshape(raw[i:j].shape[0], -1)
                dist = dtw.dtw(query, template, keep_internals=True).normalizedDistance
                if dist < threshold:
                    break
        if j != len(pred)-1:
            print("detacted : from ", i * 33, " ms to ", j * 33, " ms frame distance : ", dist, " diff ", j * 33 - i *33)
        last_task = i


print("d")
