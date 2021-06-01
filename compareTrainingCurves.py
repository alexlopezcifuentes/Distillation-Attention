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

TeacherPath = os.path.join(ResultsPath, 'Teacher ResNet50 ADE20K')
BaselinePath = os.path.join(ResultsPath, 'Baseline 3 ResNet18 ADE20K')
StudentPath = os.path.join(ResultsPath, '42 ResNet18 ADE20K')
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

epochs = 60
epc = np.arange(0, epochs)

# Accuracies
plt.figure()
plt.plot(epc, teacher_val_acc[:epochs], label='Teacher')
plt.plot(epc, bs_val_acc[:epochs], label='Baseline')
plt.plot(epc, st_val_acc[:epochs], label='Student')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves Comparisson')
plt.savefig(os.path.join(StudentPath, 'Images', 'Comparisson Accuracies.png'), dpi=300)

# Losses
plt.figure()
plt.plot(epc, teacher_loss[:epochs], label='Teacher')
plt.plot(epc, bs_loss[:epochs], label='Baseline')
plt.plot(epc, st_loss[:epochs], label='Student')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves Comparisson')
plt.savefig(os.path.join(StudentPath, 'Images', 'Comparissson Loss.png'), dpi=300)