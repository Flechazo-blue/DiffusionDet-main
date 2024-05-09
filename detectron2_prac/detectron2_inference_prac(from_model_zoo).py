"""
-*- coding: uft-8 -*-
@Description:
@Time: 2024/4/24 18:26
@Author: shaowei
"""
# detectron2相关的库
import detectron2
import cv2
from detectron2 import model_zoo  # 前人训好的模型
from detectron2.engine import DefaultPredictor  # 默认预测器
from detectron2.config import get_cfg  # 配置函数
from detectron2.utils.visualizer import Visualizer  # 可视化检测出来的框框函数
from detectron2.data import MetadataCatalog  # detectron2对数据集预留的标签
from matplotlib import pyplot as plt  # 画画函数

cfg = get_cfg()
# cfg.MODEL.DEVICE='cpu'  #如果你的电脑没有Nvidia的显卡或者你下载的是cpu版本的pytorch就将注释打开

im = cv2.imread("1.jpg")  # 放入图片的地址
# plt.figure(figsize=(20,10))
# plt.imshow(im[:,:,::-1])
# plt.show()

# 物件辨识
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # 设定参数档/迁移
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # 真正参数档
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs['instances'].pred_classes)
print(outputs['instances'].pred_boxes)  # 方框标记值

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to("cpu"))
plt.figure(figsize=(20, 10))
plt.imshow(v.get_image())
plt.show()
