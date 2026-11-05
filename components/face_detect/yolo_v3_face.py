import os
import cv2
import numpy as np
import time
import torch
from face_detect.yolov3 import Yolov3, Yolov3Tiny
from face_detect.utils.torch_utils import select_device
from face_detect.utils.utils import process_data, plot_one_box, non_max_suppression, scale_coords


# 人脸检测过程的实现
class yolo_v3_face_model(object):
    # 初始化
    def __init__(self,
                 model_path='./components/face_detect/weights/face_yolo_416-20210418.pt',
                 model_arch='yolov3',
                 yolo_anchor_scale=1.,
                 img_size=416,
                 conf_thres=0.4,
                 nms_thres=0.4, ):
        """
        :param model_path: 模型权重存储位置
        :param model_arch: 模型类别：yolo yolo-tiny
        :param yolo_anchor_scale: anchor的比例
        :param img_size: 送入网络中图像的大小
        :param conf_thres: 置信度阈值
        :param nms_thres: nms iou的阈值
        """
        # 获取设备信息
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        # 属性赋值
        # 图像大小
        self.img_size = img_size
        # 类别信息
        self.classes = ["Face"]
        # 类别个数
        self.num_classes = len(self.classes)
        # 阈值信息
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # 模型实例化
        weights = model_path
        if "tiny" in model_arch:
            # 图像的大小影响anchor的变化，根据输入图像的大小对图像进行调整
            a_scalse = 416. / img_size * yolo_anchor_scale
            anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
            anchors_new = [(int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse)) for j in range(len(anchors))]
            # 模型架构实例化
            model = Yolov3Tiny(self.num_classes, anchors=anchors_new)
        else:
            # 图像的大小影响anchor的变化，根据输入图像的大小对图像进行调整
            a_scalse = 416. / img_size * yolo_anchor_scale
            anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
            anchors_new = [(int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse)) for j in range(len(anchors))]
            # 模型架构实例化
            model = Yolov3(self.num_classes, anchors=anchors_new)
        self.model = model
        # 设备选择
        self.device = select_device()
        self.use_cuda = torch.cuda.is_available()
        # 加载权重
        if os.access(weights, os.F_OK):  # 判断模型文件是否存在
            self.model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage)['model'])
        else:
            print('------- >>> error : model not exists')
            return False
        # 模型设置为 eval，并写入到设备中
        self.model.eval()
        self.model = self.model.to(self.device)

    # 模型预测
    def predict(self, img_, vis):
        with torch.no_grad():
            # 图像预处理：图像大小调整，归一化，通道，类型等
            img = process_data(img_, self.img_size)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            # 图片检测
            pred, _ = self.model(img)
            # NMS 去除冗余，获取检测结果
            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]  # nms
            # 若检测结果为空，返回空类别
            if (detections is None) or len(detections) == 0:
                return []
            # 检测框是相对于416*416的图像的，调整检测框使其适应于原图像
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img_.shape).round()
            # 获取最终的检测结果
            output_dict_ = []
            # 遍历得到坐标，置信度，类别概率和类别信息
            for *xyxy, conf, cls_conf, cls in detections:
                label = '%s %.2f' % (self.classes[0], conf)
                x1, y1, x2, y2 = xyxy
                # 坐标信息和置信度
                output_dict_.append((float(x1), float(y1), float(x2), float(y2), float(conf.item())))
                if vis:
                    plot_one_box(xyxy, img_, label=label, color=(0, 175, 255), line_thickness=2)
            return output_dict_
