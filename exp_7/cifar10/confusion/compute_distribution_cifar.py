#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import torch


# In[2]:


import pickle


# In[4]:

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def distance(data1, data2):
    return np.linalg.norm(data1 - data2)

def get_confusion(predict, labels):
    confusion = {}
    for l1 in range(10):
        confusion[(l1, l1)] = 0
        for l2 in range(10):
            if l1 == l2 or ((l1, l2) in confusion):
                continue
            c = 0
            subcount = 0
            for i in range(len(predict)):

                if l1 == labels[i] and l2 == predict[i]:
                    c = c + 1

                if l1 == labels[i]:
                    subcount = subcount + 1

            confusion[(l1, l2)] = c/subcount
    return confusion

original_labels = np.load("original_labels.npy")
original_outputs = np.load("original_outputs.npy")
fixed_labels = np.load("fixed_labels.npy")
fixed_outputs = np.load("fixed_outputs.npy")

original_train_labels = np.load("original_train_labels.npy")
original_train_outputs = np.load("original_train_outputs.npy")
fixed_train_labels = np.load("fixed_train_labels.npy")
fixed_train_outputs = np.load("fixed_train_outputs.npy")



'''
for d in tqdm(original[5]):
    j = 0
    temp_dist = -1
    for i in range(len(original_train_outputs)):
        dist = distance(d, original_train_outputs[i])
        if (dist > 0) and (dist < temp_dist or temp_dist == -1):
            temp_dist = dist
            j = i
    original_bar[original_train_predict[j]] += 1

'''


# nearest neighbors
'''
ratios = ["0.2", "0.4", "0.6", "0.8", "1"]
fixed_bar = []
for r in ratios:
    fixed_bar.append([0 for _ in range(10)])
    fixed_train_labels = np.load("fixed_" + r + "_train_labels.npy")
    fixed_train_outputs = np.load("fixed_" + r + "_train_outputs.npy")
    fixed_labels = np.load("fixed_" + r + "_labels.npy")
    fixed_outputs = np.load("fixed_" + r + "_outputs.npy")
    fixed_train_predict = np.argmax(fixed_train_outputs, axis=1)

    fixed = [[] for _ in range(10)]
    for (output, label) in zip(fixed_outputs, fixed_labels):
        fixed[label].append(output)

    fixed_vectors = []
    for i in range(10):
        fixed_vectors.append(np.mean(fixed[i], axis=0))
    #fixed_vectors = np.mean(fixed, axis=1)


    fixed_nn = []
    temp_dist = -1
    for d in tqdm(fixed[5]):
        j = 0
        temp_dist = -1
        for i in range(len(fixed_train_outputs)):
            dist = distance(d, fixed_train_outputs[i])
            if (dist > 0) and (dist < temp_dist or temp_dist == -1):
                temp_dist = dist
                j = i
        fixed_bar[-1][fixed_train_predict[j]] += 1

    print(fixed_bar[-1])
    X = np.arange(10)

plt.bar(X + 0.00, original_bar, width = 0.1)
plt.bar(X + 0.10, fixed_bar[0], width = 0.1)
plt.bar(X + 0.20, fixed_bar[1], width = 0.1)
plt.bar(X + 0.30, fixed_bar[2], width = 0.1)
plt.bar(X + 0.40, fixed_bar[3], width = 0.1)
plt.bar(X + 0.50, fixed_bar[4], width = 0.1)
plt.legend(labels=['Original', 'fixed_0.2', 'fixed_0.4', 'fixed_0.6', 'fixed_0.8', 'fixed_1'])
plt.title("Distribution of dogs(5)'s closest neighbors")
plt.xlabel("labels")
plt.ylabel("number of data points")
plt.xticks(X, X)
plt.show()
'''



confusion_bar = []
models = ["original", "oversampling", "dbr", "softmax", "newbn"]
# confusion
for m in models:
    labels = np.load("cifar10_" + m + "_labels.npy")
    yhats = np.load("cifar10_" + m + "_yhats.npy")
    fixed_confusion = get_confusion(yhats, labels)
    confusion_bar.append([0 for _ in range(10)])
    for i in range(10):
        confusion_bar[-1][i] = fixed_confusion[(3, i)]
    print(confusion_bar[-1])
X = np.arange(10)
plt.figure(figsize=(10,6), dpi=200)
plt.bar(X + 0.00, confusion_bar[0], width = 0.15)
plt.bar(X + 0.15, confusion_bar[1], width = 0.15)
plt.bar(X + 0.30, confusion_bar[2], width = 0.15)
plt.bar(X + 0.45, confusion_bar[3], width = 0.15)
plt.bar(X + 0.60, confusion_bar[4], width = 0.15)

plt.legend(labels=['orig', 'w-aug(5)', 'w-dbr(0.5)', 'w-os(0.001)', 'w-bn(0.4)'], loc='best', framealpha=0.5, prop={'size': 16})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.gcf().subplots_adjust(left=0.2, bottom=0.15)

plt.title("Misclassification cats(3) -> others", fontsize=25)
plt.xlabel("labels", fontsize=22)
plt.ylabel("Confusion", fontsize=22)
plt.xticks(X, X)
plt.savefig('cifar10_3.pdf',bbox_inches='tight', dpi=1200, format='pdf')
plt.show()
