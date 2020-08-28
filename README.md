# RetinaNet
this is unofficial implement of retinanet.we trained this model on 4 GPUs with batch_size=32(8 image per node).the total
epoch is 18, we use adam optimizer with cosine learn rate decay.the input image_size is 640 pix(different from the paper),
first we keep the width/height ratio, resize the long side to be 640 and pad the short side.
the performance on coco is 32.3(mAp)
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