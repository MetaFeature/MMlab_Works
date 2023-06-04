# Prepare/准备
Configuration Environment \
配置环境(建议使用清华源：-i https://pypi.tuna.tsinghua.edu.cn/simple) \
```
conda create -n mmpose0524
conda activate mmpose0524
git clone https://github.com/TommyZihao/MMPose_Tutorials
#The tutorial in 2023/0524 is wonderful!
#2023/0524中的教程很精彩！
pip install -r requirements-mmpose.txt
git clone 
git clone https://github.com/open-mmlab/mmpose.git -b tutorial2023
cd mmpose
mim install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection.git -b 3.x
cd mmdetection
pip install -v -e .
cd ..
git clone https://github.com/open-mmlab/mmdeploy.git
cd ../mmpose
```
Then, put the dataset and configuration file into mmpose/data.
然后，将数据集、配置文件放入 mmpose/data 中。

# Retrain object detection/重新训练目标检测
Retrain the rtmdet:   
重新训练rtmdet：   
```
python tools/train.py data/rtmdet_tiny_ear.py
```

Evaluate object detection performance of rtmdet:  
评估rtmdet目标检测性能：  
```
python tools/test.py data/rtmdet_tiny_ear.py \
                      work_dirs/rtmdet_tiny_ear/epoch_200.pth
```
Evaluation result:  
评估结果：  
```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.08s).
Accumulating evaluation results...
DONE (t=0.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.787
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.965
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.965
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.787
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.817
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.817
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.817
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
06/03 22:12:23 - mmengine - INFO - bbox_mAP_copypaste: 0.787 0.965 0.965 -1.000 -1.000 0.787
06/03 22:12:23 - mmengine - INFO - Epoch(test) [11/11]    coco/bbox_mAP: 0.7870  coco/bbox_mAP_50: 0.9650  
        coco/bbox_mAP_75: 0.9650  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  
        coco/bbox_mAP_l: 0.7870  data_time: 0.2298  time: 0.2518
```


# Convert model/精简模型
Simplify the model with mmdetection:  
借助mmdetection中的代码精简模型：  
```
python  ../mmdetection/tools/model_converters/publish_model.py \
        work_dirs/rtmdet_tiny_ear/epoch_200.pth \
        checkpoint/rtmdet_tiny_ear_epoch_200_202306030847.pth
```

# Retrain keypoint detection/重新训练关键点检测
Retrain rtmpose:  
重新训练rtmpose:  
```
python tools/train.py data/rtmpose-s-ear.py
```

Evaluate keypoint detection performance of rtmpose:  
评估rtmpose关键点检测性能：  
```
python tools/test.py data/rtmpose-s-ear.py \
                      work_dirs/rtmpose-s-ear/epoch_300.pth
```

Evaluation result:  
评估结果：  
```
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.729
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.969
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.776
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.776
06/04 08:42:40 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
06/04 08:42:40 - mmengine - INFO - Evaluating AUC...
06/04 08:42:40 - mmengine - INFO - Evaluating NME...
06/04 08:42:40 - mmengine - INFO - Epoch(test) [6/6]    coco/AP: 0.729017  coco/AP .5: 1.000000  
        coco/AP .75: 0.969118  coco/AP (M): -1.000000  coco/AP (L): 0.729017  coco/AR: 0.776190  
        coco/AR .5: 1.000000  coco/AR .75: 0.976190  coco/AR (M): -1.000000  
        coco/AR (L): 0.776190  PCK: 0.972789  AUC: 0.138946  NME: 0.040046  
        data_time: 0.720355  time: 0.737166
```

# Convert model/精简模型
Simplify the model with mmpose：  
借助mmpose精简模型：  
```
python  tools/misc/publish_model.py \
        work_dirs/rtmpose-s-ear/epoch_300.pth \
        checkpoint/rtmpose-s-ear-300-202306030847.pth
```

# Inference/推理
pictures:  
图片：  
```
python demo/topdown_demo_with_mmdet.py \
        data/rtmdet_tiny_ear.py \
        checkpoint/rtmdet_tiny_ear_epoch_200_202306030847-87e8f8dc.pth\
        data/rtmpose-s-ear.py \
        checkpoint/rtmpose-s-ear-300-202306030847-4b6cc77d_20230604.pth \
        --input data/Ear210_Keypoint_Dataset_coco/images/DSC_5619.jpg\
        --output-root outputs/G2_RTMDet-RTMPose-Ear \
        --device cuda:0 \
        --bbox-thr 0.5 \
        --kpt-thr 0.5 \
        --nms-thr 0.3 \
        --radius 36 \
        --thickness 30 \
        --draw-bbox \
        --draw-heatmap \
        --show-kpt-idx        
```
![jpg](EarDetected_DSC_5619.jpg)

video：  
视频：  
```
python demo/topdown_demo_with_mmdet.py \
        data/rtmdet_tiny_ear.py \
        checkpoint/rtmdet_tiny_ear_epoch_200_202306030847-87e8f8dc.pth \
        data/rtmpose-s-ear.py \
        checkpoint/rtmpose-s-ear-300-202306030847-4b6cc77d_20230604.pth \
        --input data/test/MyEar.mp4 \
        --output-root outputs/G2_Ear_Video \
        --device cuda:0 \
        --bbox-thr 0.5 \
        --kpt-thr 0.5 \
        --nms-thr 0.3 \
        --radius 16 \
        --thickness 10 \
        --draw-bbox \
        --draw-heatmap \
        --show-kpt-idx
```
![gif](MyEarDetected.gif)


# Deploy ONNX/部署ONNX
```
cd ../mmdeploy
python tools/deploy.py \
        configs/mmdet/detection/detection_onnxruntime_dynamic.py \
        ../mmdetection/data/rtmdet_tiny_ear.py \
        ../mmdetection/checkpoint/rtmdet_tiny_ear_epoch_200_202306030847-87e8f8dc.pth \
        ../mmpose/data/Ear210_Keypoint_Dataset_coco/images/DSC_5619.jpg \
        --work-dir ../rtmdet2onnx \
        --dump-info
```
```
python tools/deploy.py \
        configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
        ../mmpose/data/rtmpose-s-ear.py \
        ../mmpose/checkpoint/rtmpose-s-ear-300-202306030847-3cd02a8f.pth \
        ../mmpose/data/Ear210_Keypoint_Dataset_coco/images/DSC_5619.jpg \
        --work-dir ../rtmpose2onnx \
        --dump-info
```
