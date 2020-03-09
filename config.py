video_path = 'test.mov'  # 大符视频路径
params_path = 'parameters.pkl'  # 参数文件路径
roi_range = 10  # ROI图像需扩大的像素值，如果不能正确识别R请适当修改
pic_num = 10  # 获取有效R图片的阈值，图片数量超过阈值则开始计算中心点坐标
shape = 320, 320  # 输入图片形状，请勿修改！！！
cord_error_min = 5  # 坐标误差值最小值
cord_error_max = 50  # 坐标误差最大值
area_thres = 500  # 轮廓面积阈值
img_process_params = {  # 图片处理参数,如果图片预处理效果不理想，请适当修改参数
    'green_channel_weight': 1,  # 绿色通道权重
    'blue_channel_weight': -0.9,  # 蓝色通道权重
    'threshold_min': 25,  # 二值化最小值
    'threshold_max': 255,  # 二值化最大值
    'element_size': (2, 2),  # 腐蚀操作内核大小
    'dilate_iterations': 1  # 腐蚀操作迭代次数
}