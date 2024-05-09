"""
-*- coding: uft-8 -*-
@Description:
@Time: 2024/4/25 20:22
@Author: shaowei
"""
from detectron2.data import transforms as T
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer


# 1.定义增强方法序列
augs = T.AugmentationList([
    T.RandomBrightness(0.9, 1.1),
    T.RandomFlip(prob=0.5),
    T.RandomCrop("absolute", (640, 640)),
    T.RandomRotation(angle=[20, 180])
])  # type: T.Augmentation

image = cv2.imread("./1.jpg")
boxes = np.array([[12, 12, 340, 340], [0, 0, 330, 660]])

# 2.定义输入：将图像、标签打包
input = T.AugInput(image, boxes=boxes)

# 3.执行数据增强（返回值是实际增强数值（例如旋转了多少度））
transform = augs(input)  # type: T.Transform

# 4.返回结果
image_transformed = input.image  # new image
boxes = input.boxes  # new semantic segmentation

# T.Transform可以将相同参数的变换处理其他图像:
image2_transformed = transform.apply_image(image)
polygons_transformed = transform.apply_box(boxes)

cv2.imshow("1", image_transformed)
cv2.imshow("2", image2_transformed)
cv2.waitKey(0)