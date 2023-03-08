import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(img, name):
    cv2.namedWindow(name, 2)  #创建一个窗口
    cv2.imshow(name, img)


#? 求面积函数
def rice_area(img):
    # 导入图片，图片放在程序所在目录
    img = cv2.imread(img)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用自适应阈值操作进行图像二值化
    dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 101, 1)
    # 形态学去噪
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 开运算去噪（先腐蚀再膨胀）
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element,
                           iterations=2)  # 开运算3次可以全部分开，但是面积平均值少了10px

    cv2.namedWindow("dst11", 2)  #创建一个窗口
    cv2.imshow("dst", dst)  #显示灰度图

    # 轮廓检测函数
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)

    count = 0  # 米粒总数
    ares_avrg = 0  # 米粒平均
    # 遍历找到的所有米粒
    for cont in contours:
        # 计算包围性状的面积
        ares = cv2.contourArea(cont)
        # 过滤面积小于50的形状
        if ares < 50:
            continue
        count += 1
        ares_avrg += ares
        # 打印出每个米粒的面积
        print("{}-blob:{}".format(count, ares), end="  ")
        # 提取矩形坐标
        rect = cv2.boundingRect(cont)
        # 打印坐标
        print("x:{} y:{}".format(rect[0], rect[1]))
        # 绘制矩形
        cv2.rectangle(img, rect, (0, 0, 255), 1)
        # 防止编号到图片之外（上面）,因为绘制编号写在左上角，所以让最上面的米粒的y小于10的变为10个像素
        y = 10 if rect[1] < 10 else rect[1]
        # 在米粒左上角写上编号
        cv2.putText(img, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX,
                    0.4, (0, 255, 0), 1)
        # print('编号坐标：',rect[0],' ', y)
    print('个数', count, ' 总面积', ares_avrg, ' ares', ares)
    print("米粒平均面积:{}".format(round(ares_avrg / count, 2)))  #打印出每个米粒的面积

    cv2.namedWindow("imgshow", 2)  #创建一个窗口
    cv2.imshow('imgshow', img)  #显示原始图片（添加了外接矩形）

    cv2.namedWindow("dst", 2)  #创建一个窗口
    cv2.imshow("dst", dst)  #显示灰度图

    cv2.waitKey()


#? 分水岭算法优化米粒分割
img = cv2.imread('./assets/test4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ishow = img.copy()
#! 使用局部阈值的大津算法进行图像二值化
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 101, 1)

# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
# cv2.imshow("opening",opening)
# opening = cv2.erode(opening,element,iterations=1)# 偷偷执行一次腐蚀操作

ret, thresh = cv2.threshold(opening, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("thresh",thresh)
kernel = np.ones((3, 3), np.uint8)
#* 优化
thresh = cv2.dilate(thresh, kernel, iterations=1)

# cv2.imshow("thresh2",thresh)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
sure_bg = cv2.dilate(opening, kernel, iterations=2)
#* 优化bg
sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_CLOSE, kernel, iterations=1)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
ret, sure_fg = cv2.threshold(dist_transform, 0, 255, 0)
sure_fg = np.uint8(sure_fg)
#* 优化fg
# sure_fg = cv2.dilate(sure_fg,kernel,iterations=1)
# sure_fg = cv2.erode(sure_fg,kernel,iterations=1)
# sure_fg = cv2.morphologyEx(sure_fg,cv2.MORPH_OPEN,kernel,iterations=2)

unknown = cv2.subtract(sure_bg, sure_fg)
# cv2.imshow("sure_bg",sure_bg)
# cv2.imshow("sure_fg",sure_fg)
# unknown = np.uint8(unknown)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 0]

retval, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
""" cv2.namedWindow("ishow", 2)   #创建一个窗口
cv2.imshow('ishow', ishow)    #显示原始图片
cv2.namedWindow("img", 2)   #创建一个窗口
cv2.imshow("img", img)  #显示灰度图
cv2.waitKey() """
print(result.shape)  #分水岭算法后边缘也会出现一个边框（裁剪一像素）
img = img[1:427, 1:427]
result = result[1:427, 1:427]
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite('rice_result.png', img)
cv2.imwrite('result.png', result)
#? 调用函数算面积
rice_area('rice_result.png')