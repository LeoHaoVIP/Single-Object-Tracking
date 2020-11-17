# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from numba import jit

np.set_printoptions(threshold=np.inf)
# 采用的直方图类型
HIST_TYPE = 'GRAY'
# mean-shift最大迭代次数
MAX_ITER_NUM = 10
# HSV直方图下，特征提取的通道（0-H, 1-S, 2-V）
HSV_CHANNEL = 2


# -------→ Col/x/width
# |
# |
# |
# ↓
# Row/y/height

# 加载图像数据集
@jit
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
@jit
def crop_image(_window, _image):
    [col, row, width, height] = _window
    return _image[row:row + height, col:col + width]


# 在指定图像上画框
@jit
def add_rectangle(_window, _image):
    [col, row, width, height] = _window
    return cv2.rectangle(np.copy(_image), (col, row), (col + width, row + height), 255, 2)


# Epanechnikov kernel
@jit
def e_kernel(x):
    if x <= 1:
        return 1 - x
    else:
        return 0


# 获取指定窗口各个像素点的权值(固定的)
# 对于mean_shift的优化，认为距离中心点越近的像素，对应的影响越大
@jit
def get_image_weights(width, height):
    # 目标中心点
    x0, y0 = int(height / 2), int(width / 2)
    # 核函数窗口大小
    h = x0 ** 2 + y0 ** 2
    # 各个像素点的权值
    weight = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            # 每个像素点的归一化像素位置
            pos_normed = ((i - x0) ** 2 + (j - y0) ** 2) / h
            weight[i, j] = e_kernel(pos_normed)
    return weight


# 根据像素点位置获取直方图下标索引index
@jit
def get_hist_index(i, j, _object):
    if HIST_TYPE == 'BIN':
        # 注意OpenCv读取图像时，通道顺序为BGR
        _R = np.fix(_object[i, j, 2] / 16)
        _G = np.fix(_object[i, j, 1] / 16)
        _B = np.fix(_object[i, j, 0] / 16)
        # RGB像素值结合（范围0-4095）
        index = int(_R * 256 + _G * 16 + _B)
    elif HIST_TYPE == 'HSV':
        cvt = cv2.cvtColor(np.copy(_object), cv2.COLOR_BGR2HSV)
        # 当选择HSV直方图时，需要提供参数channel，代表选择的是H、S or V
        # index = cvt[i, j, 2]
        index = min(cvt[i, j, HSV_CHANNEL], 180)
    elif HIST_TYPE == 'GRAY':
        cvt = cv2.cvtColor(np.copy(_object), cv2.COLOR_BGR2GRAY)
        index = cvt[i, j]
    else:
        index = 0
    return index


# 获取指定图像的概率直方图，直方图类型有三种：
# BIN-将RGB颜色空间量化为16x16x16
# HSV-将RGB颜色空间转化内HSV，取H通道[0,180]
# GRAY-灰度颜色空间：255*255*255
@jit
def img2prob_histogram(_object):
    [m, n, _] = np.shape(_object)
    # 各个像素点的权重（固定的）
    _weight = get_image_weights(n, m)
    if HIST_TYPE == 'BIN':
        # 直方图大小
        hist_size = 16 * 16 * 16
        _histogram = np.zeros(hist_size, dtype=np.float32)
        for i in range(m):
            for j in range(n):
                # 直方图下标
                hist_index = get_hist_index(i, j, _object)
                _histogram[hist_index] += _weight[i, j]
    elif HIST_TYPE == 'HSV' or HIST_TYPE == 'GRAY':
        hist_size = 256
        _histogram = np.zeros(hist_size, dtype=np.float32)
        for i in range(m):
            for j in range(n):
                # 直方图下标
                hist_index = get_hist_index(i, j, _object)
                _histogram[hist_index] += _weight[i, j]
    else:
        _histogram = None
    # 直方图归一化
    return _histogram / np.sum(_weight)


# 计算直方图各个bin的权重wi
@jit
def get_hist_bin_weights(h1, h2):
    hist_size = len(h1)
    sim = np.zeros(hist_size, np.float32)
    for i in range(hist_size):
        if h2[i] != 0:
            sim[i] = np.sqrt(h1[i] / h2[i])
    return sim


# 计算两个直方图的相似度（巴氏系数B）
# 直方图之间的距离D=sqrt(1-B)
@jit
def get_hist_similarity(h1, h2):
    hist_size = len(h1)
    dist = 0
    for i in range(hist_size):
        dist += np.sqrt(h1[i] * h2[i])
    return dist


# 运行mean-shift，根据目标在上一帧的位置以及目标直方图，预测在当前帧的位置
@jit
def predict_window(prior_window, target_hist, current_frame):
    # 当前帧目标位置
    current_window = np.copy(prior_window)
    [_, _, width, height] = current_window
    # 标记迭代次数
    iter_num = 0
    x_shift_old, y_shift_old = 0, 0
    while True:
        # 当前目标区域
        img_object = crop_image(current_window, current_frame)
        # 计算基于前一帧目标窗口的概率直方图
        current_hist = img2prob_histogram(img_object)
        # 直方图各个bin的权重
        bin_weights = get_hist_bin_weights(target_hist, current_hist)
        bin_weights /= np.sum(bin_weights)
        sum_weight = 0
        # mean-shift偏移向量
        x_shift, y_shift = 0, 0
        x0 = int(height / 2)
        y0 = int(width / 2)
        # 计算基于巴氏距离最小化/巴氏系数最大化的mean-shift偏移向量
        for i in range(height):
            for j in range(width):
                hist_index = get_hist_index(i, j, img_object)
                sum_weight += bin_weights[hist_index]
                x_shift += bin_weights[hist_index] * (j - y0)
                y_shift += bin_weights[hist_index] * (i - x0)
        x_shift /= sum_weight
        y_shift /= sum_weight
        # 防止出界，设置起始点的最小值为0
        current_window[0] = max(current_window[0] + x_shift, 0)
        current_window[1] = max(current_window[1] + y_shift, 0)
        if iter_num >= MAX_ITER_NUM:
            # 迭代次数达到上限，说明目标极有可能丢失，此时目标窗口不变，等待目标重新出现
            # 这里没有用到巴氏距离，原因是为了防止迭代次数过多
            current_window = prior_window
            break
        if abs(x_shift - x_shift_old) < 1e-6 and abs(y_shift - y_shift_old) < 1e-6:
            break
        x_shift_old, y_shift_old = x_shift, y_shift
        iter_num += 1
        print('window shift vector: ', [x_shift, y_shift])
        print('iterations: ', iter_num)
    return current_window


# 执行目标跟踪
@jit
def run_object_detection(first_window, _frames):
    # 目标图像概率直方图
    target_hist = img2prob_histogram(crop_image(first_window, _frames[0]))
    # 直接输出第一张图像的检测结果
    cv2.imshow('', add_rectangle(first_window, _frames[0]))
    cv2.waitKey(1)
    prior_window = first_window
    for i in range(1, len(_frames)):
        # 预测目标在当前帧的位置
        prior_window = predict_window(prior_window, target_hist, _frames[i])
        print(prior_window)
        cv2.imshow('', add_rectangle(prior_window, _frames[i]))
        cv2.waitKey(1)


if __name__ == '__main__':
    # 加载视频帧
    positions, frames, best_frames = load_data_set('./dataset/dog')
    # 设置目标起始位置
    object_first_window = positions[0]
    # object_first_window = cv2.selectROI('ROI', frames[0])
    # 执行目标检测
    run_object_detection(object_first_window, frames)
