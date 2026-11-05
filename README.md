# dpcas — 人脸识别刷脸支付 Demo

基于 PyTorch 的人脸检测 + 人脸对齐 + 人脸识别完整流程 Demo，实现对摄像头画面中的人脸进行检测、关键点/姿态矫正、特征提取与 FaceBank 比对，用于本地「刷脸支付」演示。

## 目录结构

```
main.py                       # 入口：加载配置并启动刷脸支付应用
applications/
  FacePay_local_app.py        # 刷脸支付主流程（检测 → 矫正 → 识别 → 匹配）
components/
  face_detect/                # YOLOv3 人脸检测
  face_detect_V5/             # YOLOv5 人脸检测（含 models/ utils/）
  face_euler_angle/           # 人脸欧拉角（姿态）估计，ResNet18
  face_multi_task/            # 人脸多任务网络（关键点等），ResNet34
  insight_face/               # 人脸特征提取与比对（Verification）
lib/
  facepay_lib/
    cfg/facepay.cfg           # 模型路径与阈值配置
    cores/facepay_fuction.py  # 业务逻辑
    goods/                    # 商品图片（演示用）
    make_facebank_tools/      # 构建 FaceBank（人脸底库）工具
    utils/utils.py
```

## 运行

```bash
pip install torch torchvision opencv-python numpy
python main.py
```

`main.py` 中 `video_path=0` 表示使用默认摄像头，可改为视频文件路径。

## 模型文件说明

模型权重体积较大（最大约 438 MB），**未包含在此仓库中**（GitHub 单文件上限 100 MB）。请自行下载并放入项目根目录下的 `wyw2smodels/`，路径需与 `lib/facepay_lib/cfg/facepay.cfg` 一致：

| 配置项 | 期望路径 |
| --- | --- |
| `detect_model_path` | `wyw2smodels/face_yoloV5_640.pt` |
| `face_verify_backbone_path` | `wyw2smodels/face_verify-model_ir_se-50.pth` |
| `facebank_path` | `wyw2smodels/facebank/`（含 `facebank.pth`、`names.npy`） |
| `face_multitask_model_path` | `wyw2smodels/face_multitask-resnet_34_imgsize-256-20210423.pth` |
| `face_euler_model_path` | `wyw2smodels/euler_angle-resnet_18_imgsize_256.pth` |

人脸底库可用 `lib/facepay_lib/make_facebank_tools/make_facebank.py` 由 `images/` 下的人脸照片生成。

## 备注

- 模型权重、`__pycache__`、`.DS_Store` 等已在 `.gitignore` 中排除。
- 建议后续把大模型放到 Releases 或网盘，仓库内仅保留代码与配置。
