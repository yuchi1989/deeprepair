import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import pickle
from tqdm import tqdm

def main():
    class_num = 80

    ratios = ["0.0", "0.2", "0.4", "0.6", "0.8", "1"]
    confusion_bar = []
    for r in ratios:
        if r == '0.0':
            npy_file = 'original_train_data.npy'
        else:
            npy_file = 'fix_' + r + '_train_data.npy'

        feature_data = np.load(npy_file, allow_pickle=True).item()

        object_list = feature_data["objects"]

        chosen_object = u'person'

        type2confusion = feature_data["confusion"]
        obj_conf_list = []
        conf_list = []
        for i, obj in enumerate(object_list):
            if (chosen_object, obj) in type2confusion:
                conf = type2confusion[(chosen_object, obj)]
            else:
                conf = 0
            obj_conf_list.append((obj, conf))
            conf_list.append(conf)
            print(i, obj, conf)
        print('\n'*3)

        confusion_bar.append(conf_list)


    X = np.arange(class_num)
    plt.bar(X + 0.00, confusion_bar[0], width = 0.1)
    plt.bar(X + 0.10, confusion_bar[1], width = 0.1)
    plt.bar(X + 0.20, confusion_bar[2], width = 0.1)
    plt.bar(X + 0.30, confusion_bar[3], width = 0.1)
    plt.bar(X + 0.40, confusion_bar[4], width = 0.1)
    plt.bar(X + 0.50, confusion_bar[5], width = 0.1)
    plt.legend(labels=['Original', 'fixed_0.2', 'fixed_0.4', 'fixed_0.6', 'fixed_0.8', 'fixed_1'])
    plt.title('Misclassification '+chosen_object+' -> others')
    plt.xlabel("labels")
    plt.ylabel("Confusion")
    plt.xticks(X, X)
    plt.show()


if __name__ == '__main__':
    main()
