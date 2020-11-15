# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

np.set_printoptions(threshold=np.inf)


# 加载图像数据集
def load_data_set(path):
    _positions = np.loadtxt(path + './groundtruth_rect.txt', delimiter=',', dtype=int)
    _frames = []
    _best_frames = []
    file_names = os.listdir(path)
    for i in range(len(file_names)):
        filename = file_names[i]
        if not (filename.endswith('jpg') or filename.endswith('png')):
            continue
        file_path = os.path.join(path, filename)
        img = cv2.imread(file_path)
        _frames.append(img)
        _best_frames.append(add_rectangle(_positions[i], img))
    return _positions, _frames, _best_frames


# 截取图像
def crop_image(_window, _image):
    (col, row, width, height) = _window
    return _image[row:row + height, col:col + width]


# 在指定图像上画框
def add_rectangle(_window, _image):
    (col, row, width, height) = _window
    return cv2.rectangle(np.copy(_image), (col, row), (col + width, row + height), 255, 2)


# Epanechnikov kernel
def e_kernel(x):
    if x <= 1:
        return 1 - x ** 2
    else:
        return 0


# 获取指定图像(object)各个像素点的权值
# 对于mean_shift的优化，认为距离中心点越近的像素，对应的影响越大
def get_image_weights(_window, _object):
    (col, row, width, height) = _window
    [m, n, _] = np.shape(_object)
    # 目标中心点
    x0, y0 = int(width / 2), int(height / 2)
    # 核函数窗口大小
    h = x0 ** 2 + y0 ** 2
    weight = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            # 每个像素点的归一化像素位置
            pos_normed = (((i - x0) ** 2 + (j - y0) ** 2) / h) ** 0.5
            weight[i, j] = e_kernel(pos_normed)
    return weight


# 获取指定图像的概率直方图，直方图类型有三种：
# BIN-将RGB颜色空间量化为16x16x16
# HSV-将RGB颜色空间转化内HSV，取H通道[0,180]
# FULL-原始RGB颜色空间：255*255*255
def img2prob_histogram(_window, _object, hist_type='BIN'):
    [m, n, _] = np.shape(_object)
    # 各个像素点的权重
    _weight = get_image_weights(_window, _object)
    # 直方图大小
    hist_size = 16 * 16 * 16
    _histogram = np.zeros((1, hist_size))
    for i in range(m):
        for j in range(n):
            # 注意OpenCv读取图像时，通道顺序为BGR
            _R = np.fix(_object[i, j, 2] / 16)
            _G = np.fix(_object[i, j, 1] / 16)
            _B = np.fix(_object[i, j, 0] / 16)
            # RGB像素值结合（范围0-4095）
            hist_index = int(_R * 256 + _G * 16 + _B)
            _histogram[0, hist_index] += _weight[i, j]
    # 归一化
    _histogram /= np.sum(_weight)
    return _histogram


# 运行mean-shift，预测目标出现在下一帧的位置
def predict_next_window():
    pass


# 指定目标跟踪
def run_object_detection():
    pass


if __name__ == '__main__':
    # 加载视频帧
    positions, frames, best_frames = load_data_set('./dataset/bird')
    # 设置目标起始位置
    window = tuple(positions[0])
    # 获取目标 object
    ROI = crop_image(window, frames[0])
    # 目标概率直方图
    histogram = img2prob_histogram(window, ROI)
