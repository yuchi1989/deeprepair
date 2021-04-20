import numpy as np
from sklearn.metrics import average_precision_score

# repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default='original_test_data.npy ')
parser.add_argument("--eta", type=float, default=0.8)
parser.add_argument("--eta2", type=float, default=1)
parser.add_argument("--mode", type=str, default='confusion')
args = parser.parse_args()





d = np.load(args.data_file, allow_pickle=True).item()
eta = args.eta
eta2 = args.eta2
mode = args.mode

target_class_1 = 0
target_class_2 = 5
target_class_3 = 7


features = np.concatenate(d["features"])

def sigmoid(z):
    return 1/(1 + np.exp(-z))


id2object = {
 0: u'person',
 1: u'bicycle',
 2: u'car',
 3: u'motorcycle',
 4: u'airplane',
 5: u'bus',
 6: u'train',
 7: u'truck',
 8: u'boat',
 9: u'traffic light',
 10: u'fire hydrant',
 11: u'stop sign',
 12: u'parking meter',
 13: u'bench',
 14: u'bird',
 15: u'cat',
 16: u'dog',
 17: u'horse',
 18: u'sheep',
 19: u'cow',
 20: u'elephant',
 21: u'bear',
 22: u'zebra',
 23: u'giraffe',
 24: u'backpack',
 25: u'umbrella',
 26: u'handbag',
 27: u'tie',
 28: u'suitcase',
 29: u'frisbee',
 30: u'skis',
 31: u'snowboard',
 32: u'sports ball',
 33: u'kite',
 34: u'baseball bat',
 35: u'baseball glove',
 36: u'skateboard',
 37: u'surfboard',
 38: u'tennis racket',
 39: u'bottle',
 40: u'wine glass',
 41: u'cup',
 42: u'fork',
 43: u'knife',
 44: u'spoon',
 45: u'bowl',
 46: u'banana',
 47: u'apple',
 48: u'sandwich',
 49: u'orange',
 50: u'broccoli',
 51: u'carrot',
 52: u'hot dog',
 53: u'pizza',
 54: u'donut',
 55: u'cake',
 56: u'chair',
 57: u'couch',
 58: u'potted plant',
 59: u'bed',
 60: u'dining table',
 61: u'toilet',
 62: u'tv',
 63: u'laptop',
 64: u'mouse',
 65: u'remote',
 66: u'keyboard',
 67: u'cell phone',
 68: u'microwave',
 69: u'oven',
 70: u'toaster',
 71: u'sink',
 72: u'refrigerator',
 73: u'book',
 74: u'clock',
 75: u'vase',
 76: u'scissors',
 77: u'teddy bear',
 78: u'hair drier',
 79: u'toothbrush'}


m = len(id2object)

object2id = {v:k for k,v in id2object.items()}


features = sigmoid(features)
print(features.shape)
# print(features[:2])

preds_object = []

yhats = []

for i in range(features.shape[0]):
    yhat = []
    feature = features[i]
    # confusion
    if feature[target_class_1] > 0.5 and feature[target_class_2] > 0.5:
        feature[target_class_1] *= eta
        feature[target_class_2] *= eta
    if mode == 'bias':
        if feature[target_class_1] > 0.5 and feature[target_class_3] > 0.5:
            feature[target_class_1] /= eta2
            feature[target_class_2] /= eta2

    for j, f in enumerate(feature):
        if f > 0.5:
            yhat.append(id2object[j])
    yhats.append(yhat)
    preds_object.append(feature)

preds_object = np.array(preds_object)

targets_object = np.zeros_like(features)
print('targets_object.shape', targets_object.shape)
print('preds_object.shape', preds_object.shape)
for i, label_list in enumerate(np_original['labels']):
    target_list = []
    for label in label_list:
        id = object2id[label]
        targets_object[i, id] = 1
eval_score_object = average_precision_score(targets_object, preds_object)
print('mean average precision:', eval_score_object)


type2confusion = {}
pair_count = {}
confusion_count = {}
labels = np_original['labels']
for li, yi in zip(labels, yhats):
    no_objects = [id2object[i] for i in range(m) if id2object[i] not in li]
    for i in li:
        for j in no_objects:
            if (i, j) in pair_count:
                pair_count[(i, j)] += 1
            else:
                pair_count[(i, j)] = 1

            if i in yi and j in yi:
                if (i, j) in confusion_count:
                    confusion_count[(i, j)] += 1
                else:
                    confusion_count[(i, j)] = 1
object_list = id2object.values()[1:]
for i in object_list:
    for j in object_list:
        if i == j or (i, j) not in confusion_count or pair_count[(i, j)] < 10:
            type2confusion[(i, j)] = 0
            continue
        type2confusion[(i, j)] = confusion_count[(i, j)]*1.0 / pair_count[(i, j)]

print('confusion:', type2confusion[(u'person', u'bus')], type2confusion[(u'bus', u'person')])
