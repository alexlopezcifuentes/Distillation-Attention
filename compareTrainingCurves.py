import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

compareTrainingCurves.py
Python file to plot the training val set curves for a Teacher, a Baseline and a Student Model.

Fully developed by Anonymous Code Author.
"""

ResultsPath = '/home/alc/Distillation-Attention/Results/'
Dataset = 'CIFAR100'
TeacherPath = os.path.join(ResultsPath, Dataset, 'Teachers', 'ResNet56C CIFAR100')
BaselinePath = os.path.join(ResultsPath,Dataset, 'Baselines', 'ResNet20C CIFAR100')
StudentPath = os.path.join(ResultsPath, Dataset, 'ID 5 ResNet20C CIFAR100 DFT')
Student2Path = os.path.join(ResultsPath, Dataset, 'ResNet20C CIFAR100 AT')
CurvesPath = os.path.join('Files')

with open(os.path.join(TeacherPath, CurvesPath, 'val_loss_list.pkl'), 'rb') as f:
    teacher_loss = pickle.load(f)
with open(os.path.join(TeacherPath, CurvesPath, 'val_accuracy_list.pkl'), 'rb') as f:
    teacher_val_acc = pickle.load(f)

with open(os.path.join(BaselinePath, CurvesPath, 'val_loss_list.pkl'), 'rb') as f:
    bs_loss = pickle.load(f)
with open(os.path.join(BaselinePath, CurvesPath, 'val_accuracy_list.pkl'), 'rb') as f:
    bs_val_acc = pickle.load(f)

with open(os.path.join(StudentPath, CurvesPath, 'val_loss_list.pkl'), 'rb') as f:
    st_loss = pickle.load(f)
with open(os.path.join(StudentPath, CurvesPath, 'val_accuracy_list.pkl'), 'rb') as f:
    st_val_acc = pickle.load(f)

with open(os.path.join(Student2Path, CurvesPath, 'val_loss_list.pkl'), 'rb') as f:
    st2_loss = pickle.load(f)
with open(os.path.join(Student2Path, CurvesPath, 'val_accuracy_list.pkl'), 'rb') as f:
    st2_val_acc = pickle.load(f)

epochs = len(st_loss)
epc = np.arange(0, epochs)

# Accuracies
plt.figure()
plt.plot(epc, teacher_val_acc[:epochs], label='Teacher')
plt.plot(epc, bs_val_acc[:epochs], label='Baseline')
plt.plot(epc, st_val_acc[:epochs], label='Student 1')
plt.plot(epc, st2_val_acc[:epochs], label='Student 2')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curves Comparisson')
plt.savefig(os.path.join(StudentPath, 'Images', 'Comparisson Accuracies.png'), dpi=300)

# Losses
plt.figure()
plt.plot(epc, teacher_loss[:epochs], label='Teacher')
plt.plot(epc, bs_loss[:epochs], label='Baseline')
plt.plot(epc, st_loss[:epochs], label='Student 1')
plt.plot(epc, st2_loss[:epochs], label='Student 2')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation  Loss Curves Comparisson')
plt.savefig(os.path.join(StudentPath, 'Images', 'Comparissson Loss.png'), dpi=300)