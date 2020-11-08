# COCO experiments

**Record standard outputs and files in each log_dir.**

### Train original model
```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/media/data/dataset/coco/annotations' --image_dir '/media/data/dataset/coco/'
```

### Repair confusion error
```
python2 repair_confusion.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus" --ann_dir '/media/data/dataset/coco/annotations' --image_dir '/media/data/dataset/coco/'
```
### Repair bias error
```
python2 repair_bias.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "person" --second "clock" --third "bus" --ann_dir '/media/data/dataset/coco/annotations' --image_dir '/media/data/dataset/coco/'
```
