# 人脸识别

基于 PyTorch 的人脸检测 + 人脸对齐 + 人脸识别完整流程 ，实现对摄像头画面中的人脸进行检测、关键点/姿态矫正、特征提取与 FaceBank 比对

## 运行

```bash
pip install torch torchvision opencv-python numpy
python main.py
```

`main.py` 中 `video_path=0` 表示使用默认摄像头，可改为视频文件路径。

