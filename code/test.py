# -*- coding: utf-8 -*-
#coding=utf-8

# /*
#  * @Author: kif kif101001000@163.com
#  * @Date: 2023-03-08 09:58:38
#  * @Last Modified by:   kif kif101001000@163.com
#  * @Last Modified time: 2023-03-08 09:58:38
#  */
import numpy as np
import cv2

from matplotlib import pyplot as plt


def watershed_demo(image):
    blur = cv2.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.namedWindow("binary", 2)  #创建一个窗口
    cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                          iterations=1)  #iterations1次开操作，消除图像的噪点
    cv2.namedWindow("mb", 2)  #创建一个窗口
    cv2.imshow("mb", mb)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  #3次膨胀,可以获取到大部分都是背景的区域
    cv2.namedWindow("sure_bg", 2)  #创建一个窗口
    cv2.imshow("sure_bg", sure_bg)
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist,
                                 dist.max() * 0.6, 255, cv2.THRESH_BINARY)

    surface_fg = np.uint8(
        sure_fg)  #保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    unknown = cv2.subtract(sure_bg, surface_fg)
    cv2.imshow("unkown", unknown)
    print(sure_fg[150][120:140])
    print(sure_bg[150][120:140])
    # #获取mask
    # ret, markers = cv2.connectedComponents(surface_fg)
    # markers = markers + 1
    # markers[unknown == 255] = 0
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [0, 0, 255]
    # cv2.imshow("result", img)


img = cv2.imread('./assets/coin.jpg')
watershed_demo(img)
cv2.waitKey()
