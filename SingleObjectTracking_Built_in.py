# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

# Adjustable  数据集名称 [bird、girl_in_the_sea、girl_in_the_alley、girl_in_the_garden]
DATASET_NAME = 'girl_in_the_garden'


# 加载图像数据集
def load_data_set(path):
    first_window = np.loadtxt(path + './first_window.txt', delimiter=',', dtype=int)
    _frames = []
    file_names = os.listdir(path)
    file_names.remove('first_window.txt')
    file_names.sort(key=lambda x: int(x.replace(' ', '').split('.')[0]))
    for i in range(len(file_names)):
        filename = file_names[i]
        if not (filename.endswith('jpg') or filename.endswith('png')):
            continue
        file_path = os.path.join(path, filename)
        img = cv2.imread(file_path)
        _frames.append(img)
    return first_window, _frames


# 获取数据集提供的正确检测结果
def get_best_result(positions, _frames):
    best = []
    for i in range(len(_frames)):
        fr = np.copy(_frames[i])
        [_x, _y, _w, _h] = positions[i]
        best_detect = cv2.rectangle(fr, (_x, _y), (_x + _w, _y + _h), 255, 2)
        best.append(best_detect)
    return best


if __name__ == '__main__':
    # load video
    object_first_window, frames = load_data_set('./dataset/' + DATASET_NAME)
    # set object's initial position
    (x, y, w, h) = object_first_window
    track_window = (x, y, w, h)
    # set up RoI (Region of Interest)
    roi = frames[0][y:y + h, x:x + w]
    # 计算ROI-HSV
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print(roi_hsv)
    # 计算直方图
    hist_roi = cv2.calcHist([roi_hsv], [2], None, [180], [0, 180])
    print(np.shape(hist_roi))
    # 直方图归一化
    cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
    print('----------')
    print(np.shape(hist_roi))
    # 设置终止条件，迭代10次或者至少移动1次
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 10)
    for i in range(0, len(frames)):
        print(track_window)
        frame = frames[i]
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('', img2)
        cv2.waitKey(5)
        # 计算每一帧的hsv图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 计算反向投影
        probImage = cv2.calcBackProject([hsv], [2], hist_roi, [0, 180], 1)
        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, track_window = cv2.meanShift(probImage, track_window, criteria)
