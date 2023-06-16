# 生成配置文件/build the config file
```angular2html
from mmengine import Config
cfg = Config.fromfile('../../configs/pspnet/pspnet_r50-d8_4xb2-40k_DubaiDataset.py')
cfg.data_root='Watermelon87_Semantic_Seg_Mask/'
#...#
cfg.dump('pspnet-watermelon.py')
```
[pspnet-watermelon.py](pspnet-watermelon.py)


# 训练/train
```angular2html
python tools/train.py projects/Work0616/pspnet-watermelon.py
```
Show:
```angular2html
06/16 16:05:08 - mmengine - INFO - per class results:
06/16 16:05:08 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
|    Land    | 84.42 | 99.21 |
|    Road    | 86.64 | 88.89 |
|  Building  | 31.73 | 32.61 |
| Vegetation | 53.64 |  56.1 |
|   Water    | 39.31 | 41.34 |
| Unlabeled  |  0.42 |  0.42 |
+------------+-------+-------+
06/16 16:05:08 - mmengine - INFO - Iter(val) [11/11]    aAcc: 88.4500  mIoU: 49.3600  mAcc: 53.0900  data_time: 0.0034  time: 0.0987
06/16 16:05:22 - mmengine - INFO - Iter(train) [2900/3000]  lr: 9.3518e-03  eta: 0:00:14  time: 0.1444  data_time: 0.0026  memory: 3774  loss: 0.0396  decode.loss_ce: 0.0285  decode.acc_seg: 81.8024  aux.loss_ce: 0.0111  aux.acc_seg: 81.7413
06/16 16:05:36 - mmengine - INFO - Exp name: pspnet-watermelon_20230616_155808
06/16 16:05:36 - mmengine - INFO - Iter(train) [3000/3000]  lr: 9.3294e-03  eta: 0:00:00  time: 0.1497  data_time: 0.0026  memory: 3774  loss: 0.0361  decode.loss_ce: 0.0256  decode.acc_seg: 86.1969  aux.loss_ce: 0.0106  aux.acc_seg: 87.1094
06/16 16:05:36 - mmengine - INFO - Saving checkpoint at 3000 iterations
```

#测试/test
```angular2html
python tools/test.py projects/Work0616/pspnet-watermelon.py work_dirs/pspnet-watermelon/iter_3000.pth
```
Show:
```angular2html
after_run:
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
06/16 16:06:59 - mmengine - WARNING - The prefix is not set in metric class IoUMetric.
Loads checkpoint by local backend from path: work_dirs/pspnet-watermelon/iter_3000.pth
06/16 16:06:59 - mmengine - INFO - Load checkpoint from work_dirs/pspnet-watermelon/iter_3000.pth
06/16 16:07:08 - mmengine - INFO - per class results:
06/16 16:07:08 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
|    Land    | 82.58 | 97.09 |
|    Road    | 81.81 | 85.05 |
|  Building  | 37.36 | 44.04 |
| Vegetation | 57.83 | 58.86 |
|   Water    | 45.44 | 49.25 |
| Unlabeled  |  3.62 |  3.62 |
+------------+-------+-------+
06/16 16:07:08 - mmengine - INFO - Iter(test) [11/11]    aAcc: 87.1900  mIoU: 51.4400  mAcc: 56.3200  data_time: 0.0127  time: 0.8340
```

#推理/Inference
```angular2html
python demo/image_demo.py \
        Watermelon87_Semantic_Seg_Mask/img_dir/val/R.jpeg \
        projects/Work0616/pspnet-watermelon.py \
        work_dirs/pspnet-watermelon/iter_3000.pth \
        --out-file outputs/outputs.jpg \
        --device cuda:0 \
        --opacity 0.7
```
Show:
![Work04_result.jpg](Work04_result.jpg)

