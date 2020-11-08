# COCO experiments

### Train original model
python2 train_epoch_graph.py --log_dir original_model

### Repair confusion error
python2 repair_confusion.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus"

### Repair bias error
python2 repair_bias.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "person" --second "clock" --third "bus"

