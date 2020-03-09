[TOC]
# 大符中心识别



## 快速上手

### 1. 安装依赖库

```bash
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python main.py
```

### 3. 结果说明

![result](https://github.com/BeingGod/recognizeR/blob/master/pic/result.jpg)

大符中心坐标为`(1039.0, 434.5)`，时间开销为`0.52952`秒

### 4. 调试模式

#### 调试模式说明

调试模式下将读取完整视频流，并且展示图像和打印获取中心坐标

![debug](https://github.com/BeingGod/recognizeR/blob/master/pic/debug.png)

#### 启用调试模式

修改`main.py`文件，将`debug=False`修改为`debug=True`，默认为`False`



## 原理

### 程序流程

![process](https://github.com/BeingGod/recognizeR/blob/master/pic/process.png)

### ROI获取

ROI的获取主要利用opncv库进行**图片预处理**，将图片转化为二值图。然后根据较亮轮廓寻找**最小外接矩形**，从而获取ROI区域。



### 中心识别

#### 分析

由于大符中心有确定且易识别的特征，并且需要快速获取中心坐标。将判断大符中心特征视为**二分类问题**，并利用**Logistics回归**对大符中心做判断。

#### 测试效果

对于测试集测试准确度为**99.1%**。在实际测试过程中，发现获取的ROI区域的大小对于识别准确度有较大影响，猜测是由于**网络结构浅**或者**过拟合**导致的。

#### 解决办法

目前一个解决办法是将**ROI区域范围**作为一项**可调参数**，在实际比赛中通过机器人拍摄的画面来确定其值



## 文件目录结构

```
.
├── pic								                // 图片
├── activate_func.py                  // 激活函数
├── config.py              		        // 参数配置文件
├── recognizeR.py                	    // 图像处理函数
├── main.py                    		    // 主函数
├── parameters.pkl                    // 预训练参数
├── predict.py						            // 识别函数
└── requirements.txt                  // 依赖库
```



## 优化方向

### 参数优化

调整`config.py`中参数，提升**图片预处理效果**，提高识别精度

### 加深网络

利用**加深神经网络层数**，提高神经网络**鲁棒性**



## 技术交流

|  作者  |      微信号      |
| :----: | :--------------: |
| 章睿彬 | Bgod_zhangruibin |

**如有BUG或者对程序有更好的建议欢迎提交issue或者联系作者**
