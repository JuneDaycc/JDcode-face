# -*-coding:utf-8-*-
# date:2021-04-18
# Author: Eric.Lee
# function: who you want to see "你想看谁"

import os
import cv2
import time

import numpy as np
import random
import time
import shutil

# 加载模型组件库
from components.face_detect.yolo_v3_face import yolo_v3_face_model
from components.face_detect_V5.yolo_V5_face import yolo_v5_face_model
from components.insight_face.face_verify import insight_face_model
from components.face_multi_task.face_multi_task_component import FaceMuitiTask_Model
from components.face_euler_angle.face_euler_angle_component import FaceAngle_Model
# 加载工具库
import sys

sys.path.append("./lib/facepay_lib/")
from lib.facepay_lib.cores.facepay_fuction import get_faces_batch_attribute
from lib.facepay_lib.utils.utils import parse_data_cfg, compute_face_iou


def main_facePay(video_path, cfg_file):
    """
    :param video_path: 视频位置
    :param cfg_file: 配置文件
    :return:
    """
    # 获取配置信息
    config = parse_data_cfg(cfg_file)
    for k_ in config.keys():
        print("{} : {}".format(k_, config[k_]))
    print("\n loading Face Pay local demo ...\n")
    # 加载预训练模型
    # face_detect_model = yolo_v3_face_model(conf_thres=float(config["detect_conf_thres"]),
    #                                        nms_thres=float(config["detect_nms_thres"]),
    #                                        model_arch=config["detect_model_arch"],
    #                                        model_path=config["detect_model_path"],
    #                                        yolo_anchor_scale=float(config["yolo_anchor_scale"]),
    #                                        img_size=float(config["detect_input_size"]),
    #                                        )
    face_detect_model = yolo_v5_face_model(conf_thres=float(config["detect_conf_thres"]),
                                           nms_thres=float(config["detect_nms_thres"]),
                                           weights=config["detect_model_path"],
                                           img_size=float(config["detect_input_size"]),
                                           )
    face_verify_model = insight_face_model(backbone_model_path=config["face_verify_backbone_path"],
                                           facebank_path=config["facebank_path"],
                                           threshold=float(config["face_verify_threshold"]))

    face_multitask_model = FaceMuitiTask_Model(model_path=config["face_multitask_model_path"],
                                               model_arch=config["face_multitask_model_arch"])

    face_euler_model = FaceAngle_Model(model_path=config["face_euler_model_path"])

    print("\n/------------------------------------------------------------------------/\n")
    # 随机设置不同的颜色进行绘图
    p_colors = []
    for i in range(len(face_verify_model.face_names)):
        p_colors.append((random.randint(60, 255), random.randint(70, 255), random.randint(130, 255)))
    # 遍历所有的商品
    goods_sku = []
    print("loading goods sku ...")
    for f_ in os.listdir("./lib/facepay_lib/goods/"):
        goods_sku.append("./lib/facepay_lib/goods/" + f_)
        print("sku : ", f_.replace(".jpg", ""))
    # 商品文件数据
    goods_map = np.zeros([100 * 2, 100 * 12, 3]).astype(np.uint8)
    # 购买商品列表
    goods_buy_list = []
    # 将数据进行初始化
    goods_map[:, :, :] = (200, 200, 200)
    # 交并比阈值
    face_iou_thr = 0.8
    # 是否开启人脸支付模式，实际中可能有其他的支付方式
    FacePay_Pattern = True
    # 获取视频文件流
    cap = cv2.VideoCapture(video_path)
    # 视频写入初始化
    video_writer = None
    # 帧计数
    frame_idx = 0
    # 视频流打开
    while cap.isOpened():
        # 读取帧图像
        ret, img = cap.read()
        # 读取成功
        if ret:
            # 帧计数加1
            frame_idx += 1
            # 获取副本
            algo_image = img.copy()
            # 检测人脸，获取人脸的边界框
            faces_bbox = face_detect_model.predict(img, vis=True)
            # 若检测到人脸
            if len(faces_bbox) > 0:
                # 获取确认的人脸检测结果及框，所有的检测框，人脸面积
                faces_identify, faces_identify_bboxes, bboxes, face_map = get_faces_batch_attribute(
                    face_multitask_model, face_euler_model, faces_bbox, algo_image, use_cuda=True, vis=True)

                # 绘制人脸识别有效区域，有效区域是距边界150 60的区域
                reco_edge_w = 150
                reco_edge_h = 60
                face_reco_area = (
                    reco_edge_w, reco_edge_h, algo_image.shape[1] - reco_edge_w, algo_image.shape[0] - reco_edge_h)
                cv2.rectangle(algo_image, (reco_edge_w, reco_edge_h),
                              (algo_image.shape[1] - reco_edge_w, algo_image.shape[0] - reco_edge_h), (255, 55, 55), 6)
                cv2.rectangle(algo_image, (reco_edge_w, reco_edge_h),
                              (algo_image.shape[1] - reco_edge_w, algo_image.shape[0] - reco_edge_h), (95, 95, 255), 2)
                cv2.putText(algo_image, "Face_Pay_Recognize_Area", (reco_edge_w, reco_edge_h - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 8)
                cv2.putText(algo_image, "Face_Pay_Recognize_Area", (reco_edge_w, reco_edge_h - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (250, 60, 60), 2)

                # 若人脸识别的结果数量大于0
                if len(faces_identify) > 0:
                    # 进行人脸识别：返回人脸索引及距离
                    results, face_dst = face_verify_model.predict(faces_identify)
                    face_dst = list(face_dst.cpu().detach().numpy())
                    # 遍历每一个检测结果
                    for idx, bbox_ in enumerate(faces_identify_bboxes):
                        # 计算检测结果与人脸识别有效区域之间的距离
                        face_iou_ = compute_face_iou((bbox_[1], bbox_[0], bbox_[3], bbox_[2]),
                                                     (face_reco_area[1], face_reco_area[0], face_reco_area[3],
                                                      face_reco_area[2]))
                        # 顾客进入人脸识别区域才进行识别
                        if face_iou_ > face_iou_thr:
                            # 将检测结果绘制在图像上
                            cv2.putText(algo_image, face_verify_model.face_names[results[idx] + 1] + ": {:.2f}".format(
                                face_dst[idx]), (bbox_[0], bbox_[1] + 27), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 95, 220),
                                        6)
                            cv2.putText(algo_image, face_verify_model.face_names[results[idx] + 1] + ": {:.2f}".format(
                                face_dst[idx]), (bbox_[0], bbox_[1] + 27), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                        p_colors[results[idx] + 1], 2)
            # 若没有检测结果则绘制空即可
            else:
                face_map = np.zeros([112 * 3, 112 * 3, 3]).astype(np.uint8)
                face_map[:, :, 0].fill(205)
                face_map[:, :, 1].fill(205)
                face_map[:, :, 2].fill(205)
                print(" ------ ")

            # 结果展示，绘制人脸支付的文字
            if FacePay_Pattern == True:
                cv2.putText(algo_image, "Face_Pay", (6, 36), cv2.FONT_HERSHEY_DUPLEX, 1.3, (200, 200, 200), 12)
                cv2.putText(algo_image, "Face_Pay", (6, 36), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 225, 20), 2)

            # 人脸检测结果合并显示
            face_map = cv2.resize(face_map, (algo_image.shape[0], algo_image.shape[0]))
            algo_image = np.hstack((algo_image, face_map))
            # 商品展示结果合并显示
            goods_map_r = cv2.resize(goods_map, (algo_image.shape[1], 210))
            algo_image = np.vstack((algo_image, goods_map_r))
            # 可视化设备进行展示
            cv2.namedWindow('Face_Pay', 0)
            cv2.imshow('Face_Pay', algo_image)
            # 若无可视化设备则将其写入到视频文件中
            if video_writer is None:
                loc_time = time.localtime()
                time_str = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter("recording_{}.mp4".format(time_str), fourcc, 12,
                                               (algo_image.shape[1], algo_image.shape[0]))
            video_writer.write(algo_image)
        else:
            break
        # 视频刷新时间
        key_id = cv2.waitKey(10)
        if key_id == 27:
            break
        # 模拟商品交易环节
        if len(goods_buy_list) < 23:
            # 随机购买商品
            goods_id = random.randint(0, len(goods_sku) - 1)
            # 添加商品
            goods_buy_list.append(goods_sku[goods_id])

            # 渲染需要购买的商品，获取商品图片
            img_goods = cv2.imread(goods_sku[goods_id])
            # 图像缩放
            img_goods = cv2.resize(img_goods, (100, 100))
            # 获取索引
            buy_idx = len(goods_buy_list) - 1
            # 获取渲染位置
            y1_map, y2_map = int(buy_idx / 12) * 100, (int(buy_idx / 12) + 1) * 100
            x1_map, x2_map = int(buy_idx % 12) * 100, (int(buy_idx % 12) + 1) * 100
            # 添加商品
            goods_map[y1_map:y2_map, x1_map:x2_map, :] = img_goods
            cv2.rectangle(goods_map, (int(x1_map), int(y1_map)), (int(x2_map), int(y2_map)), (55, 155, 255), 4)
            cv2.rectangle(goods_map, (int(x1_map), int(y1_map)), (int(x2_map), int(y2_map)), (255, 55, 55), 2)
    # 资源释放
    cv2.destroyAllWindows()
    video_writer.release()
