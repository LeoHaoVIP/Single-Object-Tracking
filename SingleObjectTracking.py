# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os


# 加载图像数据集
def load_data_set(path):
    _frames = []
    file_names = os.listdir(path)
    for filename in file_names:
        if not (filename.endswith('jpg') or filename.endswith('png')):
            continue
        file_path = os.path.join(path, filename)
        img = cv2.imread(file_path)
        _frames.append(img)
    object_pos_array = np.loadtxt(path + './groundtruth_rect.txt', delimiter=',', dtype=int)
    return object_pos_array, _frames


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
    pos_array, frames = load_data_set('./dataset/bird')
    # 加载正确的检测结果
    best_frames = get_best_result(pos_array, frames)
    # set object's initial position
    (x, y, w, h) = tuple(pos_array[0])
    track_window = (x, y, w, h)
    # set up RoI (Region of Interest)
    roi = frames[0][y:y + h, x:x + w]
    # 计算ROI-HSV
    # TODO: sometimes not necessary
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print(roi_hsv)
    # 去除低亮度值
    # TODO: sometimes not necessary
    mask = cv2.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    hist_roi = cv2.calcHist([roi_hsv], [2], None, [180], [0, 180])
    # 直方图归一化
    cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
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
