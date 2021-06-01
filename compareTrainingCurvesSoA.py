import pickle
import os
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

compareTrainingCurvesSoA.py
Python file to plot the training val set curves for the proposed method and the set of KD Algorithsm.
This is the file used to create the figure from the paper.

Fully developed by Anonymous Code Author.
"""


ResultsPath = 'Results/'
CurvesPath = os.path.join('Files')
Dataset = 'ADE20K'

# Models to compare
TeacherPath = {'Name': 'Teacher R50', 'Path': os.path.join('Teachers/Teacher ResNet50 ' + Dataset), 'Color': 'tab:blue'}
VanillaPath = [{'Name': 'Vanilla R18', 'Path': os.path.join('Baselines/Baseline 3 ResNet18 ' + Dataset), 'Color': 'tab:orange'}]
OursPath = [{'Name': 'DCT (Ours)', 'Path': '1 ResNet18 ' + Dataset + ' DFT', 'Color': 'tab:green'},
            {'Name': 'DCT+KD (Ours)', 'Path': '1 ResNet18 ' + Dataset + ' DFT+KD', 'Color': 'tab:green'}]
SoAPath = [{'Name': 'AT', 'Path': '1 ResNet18 ' + Dataset + ' AT', 'Color': 'tab:red'},
           {'Name': 'PKT', 'Path': '1 ResNet18 ' + Dataset + ' PKT', 'Color': 'tab:purple'},
           {'Name': 'VID', 'Path': '1 ResNet18 ' + Dataset + ' VID', 'Color': 'tab:brown'},
           {'Name': 'CRD', 'Path': '1 ResNet18 ' + Dataset + ' CRD', 'Color': 'tab:gray'},
           {'Name': 'Vanilla+KD', 'Path': os.path.join('1 ResNet18 ' + Dataset + ' KD'), 'Color': 'tab:orange'},
           {'Name': 'AT+KD', 'Path': '1 ResNet18 ' + Dataset + ' AT+KD', 'Color': 'tab:red'},
           {'Name': 'PKT+KD', 'Path': '1 ResNet18 ' + Dataset + ' PKT+KD', 'Color': 'tab:purple'},
           {'Name': 'VID+KD', 'Path': '1 ResNet18 ' + Dataset + ' VID+KD', 'Color': 'tab:brown'},
           {'Name': 'CRD+KD', 'Path': '1 ResNet18 ' + Dataset + ' CRD+KD', 'Color': 'tab:gray'}]

Methods = list()
Methods.append(TeacherPath)
Methods.extend(VanillaPath)
Methods.extend(SoAPath)
Methods.extend(OursPath)

epochs = 60
epc = np.arange(0, epochs)

plt.figure()
for method in Methods:
    with open(os.path.join(ResultsPath, Dataset, method['Path'], CurvesPath, 'val_loss_list.pkl'), 'rb') as f:
        loss = pickle.load(f)
    with open(os.path.join(ResultsPath, Dataset, method['Path'], CurvesPath, 'val_accuracy_list.pkl'), 'rb') as f:
        val_acc = pickle.load(f)

    if method['Name'].find('+KD') == -1:
        plt.plot(epc, val_acc[:epochs], color=method['Color'], label=method['Name'])
    else:
        plt.plot(epc, val_acc[:epochs], color=method['Color'], linestyle='--', label=method['Name'])

plt.legend(ncol=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('Comparison Accuracies.pdf', dpi=300)