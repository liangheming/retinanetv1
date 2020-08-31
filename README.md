# RetinaNet
This is an unofficial pytorch implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

##requirement
```text
apex
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
##result
we trained this repo on 4 GPUs with batch size 32(8 image per node).the total epoch is 18,Adam with cosine lr decay is used for optimizing.
finally, this repo achieves 32.3 mAp at 640px(long side) resolution with resnet50 backbone.
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.490
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.343
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.462
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.439
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.658
```

## difference from original implement
the main difference is about the input resolution.the original implement use *min_thresh* and *max_thresh* to keep the short
side of the input image larger than *min_thresh* while keep the long side smaller than *max_thresh*.for simplicity we fix the long
side a certain size, then we resize the input image while **keep the width/height ratio**, next we pad the short side.the final
width and height of the input are same.


## training
for now we only support coco detection data.

### COCO
* modify main.py (modify config file path)
```python
from processors.ddp_apex_processor import DDPApexProcessor
if __name__ == '__main__':
    processor = DDPApexProcessor(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: retinanet
data:
  train_annotation_path: ../annotations/instances_train2017.json 
  val_annotation_path: ../annotations/instances_val2017.json
  train_img_root: ../data/train2017
  val_img_root: ../data/val2017
  img_size: 896
  use_crowd: False
  batch_size: 4
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  anchor_sizes: [32, 64, 128, 256, 512]
  strides: [8, 16, 32, 64, 128]
  backbone: resnet18
  backbone_weight: weights/resnet18.pth
  freeze_bn: False

hyper_params:
  iou_thresh: 0.5
  ignore_thresh: 0.4
  alpha: 0.25
  gamma: 2.0
  beta: 1./9
  multi_scale: [896]

optim:
  optimizer: Adam
  lr: 0.0001
  momentum: 0.9
  milestones: [12,18]
  cosine_weights: 1.0
  warm_up_epoch: 0.
  max_norm: 2
  weight_decay: 0.0001
  epochs: 18
  sync_bn: True
val:
  interval: 1
  weight_path: weights
  conf_thresh: 0.05
  iou_thresh: 0.5
  max_det: 300

gpus: 0,1,2,3
```
* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support
