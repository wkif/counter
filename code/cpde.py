# -*- coding: utf-8 -*-
#coding=utf-8

# /*
#  * @Author: kif kif101001000@163.com
#  * @Date: 2023-03-08 09:58:38
#  * @Last Modified by:   kif kif101001000@163.com
#  * @Last Modified time: 2023-03-08 09:58:38
#  */

import cv2 as cv
import numpy as np


def watershed_demo(image):
    blur = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  #获取灰度图像

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #形态学操作，进一步消除图像中噪点
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel,
                         iterations=2)  #iterations连续两次开操作
    sure_bg = cv.dilate(mb, kernel, iterations=3)  #3次膨胀,可以获取到大部分都是背景的区域
    cv.imshow("sure_bg", sure_bg)
    #距离变换
    dist = cv.distanceTransform(mb, cv.DIST_L2, 5)
    cv.imshow("dist", dist)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    # print(mb[150][120:140])
    # print(dist[150][120:140])
    # print(dist_output[150][120:140])
    cv.imshow("distinct-t", dist_output * 50)
    ret, sure_fg = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("sure_fg", sure_fg)
    # print(sure_fg[150][120:140])
    # print(sure_bg[150][120:140])
    #获取未知区域
    surface_fg = np.uint8(
        sure_fg)  #保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    unknown = cv.subtract(sure_bg, surface_fg)
    cv.imshow("unkown", unknown)
    #获取maskers,在markers中含有种子区域
    ret, markers = cv.connectedComponents(surface_fg)
    #print(ret)

    #分水岭变换
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]

    cv.imshow("result", image)


src = cv.imread("./assets/test4.png")  #读取图片
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)  #创建GUI窗口,形式为自适应
cv.imshow("input image", src)  #通过名字将图像和窗口联系

watershed_demo(src)

cv.waitKey(0)  #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv.destroyAllWindows()  #销毁所有窗口