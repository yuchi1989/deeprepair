import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer


original = 'original_test_data.npy'
repaired = 'fix_0.8_test_data.npy'


np_original = np.load(original, allow_pickle=True).item()
features = np.concatenate(np_original["features"])
eta = 0.12

# print(len(np_original['labels']))
# print(np_original['labels'][0])
# print(len(np_original["objects"]))
# print(np_original["objects"][2])
#
#
# print(features.shape)
# print(features[200])


def sigmoid(z):
    return 1/(1 + np.exp(-z))


id2object = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

object2id = {v:k for k,v in id2object.items()}


features = sigmoid(features)
print(features.shape)
# print(features[:2])

preds_object = []

yhats = []

for i in range(features.shape[0]):
    yhat = []
    feature = features[i]
    if feature[0] > 0.5 and feature[5] > 0.5:
        feature[0] -= eta
        feature[5] -= eta
        # print(feature[0], feature[5])
        feature[1:5] += 0 #eta/39
        feature[6:] += 0 #eta/39

    for j, f in enumerate(feature):
        if f > 0.5:
            yhat.append(id2object[j+1])
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
        targets_object[i, id-1] = 1
eval_score_object = average_precision_score(targets_object, preds_object)
print('eval_score_object', eval_score_object)


type2confusion = {}
pair_count = {}
confusion_count = {}
labels = np_original['labels']
for li, yi in zip(labels, yhats):
    no_objects = [id2object[i+1] for i in range(80) if id2object[i+1] not in li]
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

print(type2confusion[(u'person', u'bus')], type2confusion[(u'bus', u'person')])
