import numpy as np
import cv2 as cv
import config  # 导入配置文件
from predict import Logistic

"""
功能要求：
    1、调用程序，输出大符圆心R中点坐标
实现流程：
    1、打开摄像头
    2、预处理图像
    3、获取ROI区域
    4、预测
    5、返回R的中心坐标
已知BUG：
    1、由于predict函数参数存在一定过拟合,因此对输入ROI区域大小敏感
"""


# 修改输入图片大小
def resizeImg(src, shape=config.shape):
    """
    :param src: 源图片
    :param shape: 目标图片大小
    :return: 展平后的图片,其形状为(shape[0]*shape[1],1)
    """
    dst = None
    if src.shape != shape:
        dst = cv.resize(src, shape, interpolation=cv.INTER_AREA)  # 将图片形状修改为指定shape

    return dst, dst.reshape(-1, 1) / 255


# 图片预处理，获取二值图
def imgProcess(src, params_dict=config.img_process_params):
    """
    :param src: 源图像
    :param params_dict: img_process_params = {  # 图片处理参数,如果图片预处理效果不理想，请适当修改参数
                            'green_channel_weight': 1,  # 绿色通道权重
                            'blue_channel_weight': -0.9,  # 蓝色通道权重
                            'threshold_min': 25,  # 二值化最小值
                            'threshold_max': 255,  # 二值化最大值
                            'element_size': (2, 2),  # 腐蚀操作内核大小
                            'dilate_iterations': 1  # 腐蚀操作迭代次数
                        }
    :return: 预处理后的图像
    """
    b, g, r = cv.split(src)
    gray = cv.addWeighted(g, params_dict['green_channel_weight'], b, params_dict['blue_channel_weight'], 0)
    thres = cv.inRange(gray, params_dict['threshold_min'], params_dict['threshold_max'])

    element = cv.getStructuringElement(1, params_dict['element_size'])
    dst = cv.dilate(thres, element, params_dict['dilate_iterations'])
    return dst, src


# 判断图片是否有R
def predict(x):
    """
    :param x: 输入值
    :return: 1或0
    """
    return Logistic().predict(x)


class Recognize:
    """
    描述：根据摄像头获取的图片返回大符中心坐标
    """
    def __init__(self, device_id, debug=False):
        self.camera_id = device_id  # 摄像头ID
        self.roi_range = config.roi_range  # ROI图像扩大像素值
        self.pic_num = config.pic_num  # 获取有效R的阈值，超过阈值则开始计算中心点坐标
        self.cord_error_min = config.cord_error_min  # 坐标误差最小值
        self.cord_error_max = config.cord_error_max  # 坐标误差最大值
        self.area_thres = config.area_thres  # 轮廓面积阈值
        self.debug = debug  # debug模式，默认不启用

    # 获取ROI区域
    def getROI(self, src, box):
        """
        :param src: 源图像
        :param box: 最小外接旋转矩形顶点坐标
        :return: ROI图像，ROI图像区域坐标
        """
        rect = cv.boundingRect(box)  # 获取最小外接矩形
        x, y, width, height = rect[0], rect[1], rect[2], rect[3]
        if 0 <= x and 0 <= width and x + width <= src.shape[1] and \
                0 <= y and 0 <= height and y + height <= src.shape[0]:
            roi = src[y - self.roi_range:y + height + self.roi_range,
                  x - self.roi_range:x + width + self.roi_range]  # 获取ROI区域
            roi_cord = (x - self.roi_range, x + width + self.roi_range,
                        y - self.roi_range, y + height + self.roi_range)  # 获取ROI区域坐标,(xmin,xmax,ymin,ymax)
            return roi, roi_cord

    # 寻找轮廓
    def findContours(self, thres, src):
        """
        :param thres: 二值图像
        :param src: 源图像
        :return: ROI图像列表，ROI图像区域坐标列表
        """
        contours, hierarchy = \
            cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)  # 寻找轮廓
        rois = []
        rois_loc = []

        for contour in contours:
            if cv.contourArea(contour) >= self.area_thres:
                min_rect = cv.minAreaRect(contour)
                points = cv.boxPoints(min_rect)  # 获取最小外接矩形的4个顶点坐标
                points = np.int0(points)

                roi, roi_cord = self.getROI(thres, points)
                rois.append(roi)
                rois_loc.append(roi_cord)

                # debug模式绘制外接矩形
                if self.debug == 1:
                    cv.drawContours(src, [points], 0, (255, 255, 255), 4)  # 绘制边框
                    cv.imshow("source", src)
                    cv.waitKey(30)

        return rois, rois_loc

    # 获取R中心坐标
    def getCenterLoc(self, loc_list):
        """
        :param loc_list: 可能的中心坐标列表
        :return: 中心坐标或0(获取失败)
        """
        cnt = 0  # 有效坐标数
        init_index = 0  # 初始坐标索引
        # 初始化坐标
        init_xmin,init_xmax,init_ymin,init_ymax =\
            loc_list[init_index][0], loc_list[init_index][1], \
            loc_list[init_index][2], loc_list[init_index][3]

        for index in range(1,len(loc_list)):
            xmin,xmax,ymin,ymax = \
                loc_list[index][0], loc_list[index][1], loc_list[index][2], loc_list[index][3]

            # xmin,xmax,ymin,ymax其中两个以上坐标误差在5以内
            if ((abs(init_xmax-xmax) <= self.cord_error_min) +
                    (abs(init_xmin-xmin) <= self.cord_error_min) +
                    (abs(init_ymax-ymax) <= self.cord_error_min) +
                    (abs(init_ymin-ymin) <= self.cord_error_min) >= 2):
                cnt += 1

            # 初始坐标xmin,xmax,ymin,ymax与当前坐标误差超过50，说明该坐标无效，将当前目标作为初始坐标索引
            elif ((abs(init_xmax-xmax) >= self.cord_error_max or
                    (abs(init_xmin-xmin) >= self.cord_error_max) or
                    (abs(init_ymax-ymax) >= self.cord_error_max) or
                    (abs(init_ymin-ymin) >= self.cord_error_max))):
                init_index = index

            # 有效坐标数>=(0.8*len(loc_list))则计算中心点坐标并返回，否则返回0，继续获取R的坐标
            if cnt >= int(0.8*len(loc_list)):
                center_loc = ((init_xmax + init_xmin)/2, (init_ymax + init_ymin)/2)
                return center_loc
        return 0

    def capture(self):
        """
        :return: 中心坐标
        """
        # 如果camera_id为-1则从视频读取，否则从摄像头读取
        if self.camera_id == -1:
            cap = cv.VideoCapture(config.video_path)
        else:
            cap = cv.VideoCapture(self.camera_id, cv.CAP_DSHOW)

        # 判断摄像头是否可用
        if not cap.isOpened():
            print("Cannot open camera")
            return 0

        center_list = []  # 可能中心坐标列表

        while True:
            # 逐帧捕获
            flag, frame = cap.read()
            # 如果正确读取帧，flag为True
            if not flag:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            thres, src = imgProcess(frame)
            rois, rois_loc = self.findContours(thres, src)

            for index in range(len(rois)):
                resize_img, x = resizeImg(rois[index])

                # debug模式显示获取ROI图像
                if self.debug == 1:
                    cv.imshow("ROI", resize_img)
                    cv.waitKey(30)

                res = predict(x)
                if res == 1:

                    # debug模式显示R图像
                    if self.debug == 1:
                        cv.imshow("R",resize_img)
                        cv.waitKey(30)

                    center_list.append(rois_loc[index])

            if len(center_list) >= self.pic_num:
                center_loc = self.getCenterLoc(center_list)

                # debug模式将读取完整视频流
                if self.debug == 1:
                    center_loc = self.getCenterLoc(center_list)
                    print("Center coordinate is {}.".format(center_loc))
                    center_list.clear()  # 释放内存

                # 若返回值不为0，说明坐标有效，返回中心坐标
                elif center_loc != 0:
                    cap.release()  # 释放摄像头
                    cv.destroyAllWindows()  # 清除创建的所有窗口
                    return center_loc

        cv.destroyAllWindows()  # 清除创建的所有窗口
        return 0
