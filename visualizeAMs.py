import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import shutil
import argparse

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

visualizeAMs.py
Python file to visualize the obtained Activation Maps with 'extractAMs.py' from a set of models. 
It has to be fed with the Model path.
In addition, you can set Teacher and Baseline Paths to obtain a comparative figure like the one
from the Paper.

The images will be divided into folders depending on the predictions from the Model.

Fully developed by Anonymous Code Author.
"""


parser = argparse.ArgumentParser(description='CAMs Comparative')
parser.add_argument('--Model', metavar='DIR', help='Folder to be evaluated', required=True)

args = parser.parse_args()
ID = args.Model

# Results where the comparative figures will be saved
ResultsPath = '/home/alc/Distillation-Attention/Results/'

# Dataset to be used
Dataset = 'ADE20K'
# Teacher to be used
Teacher = 'Teacher ResNet50 ' + Dataset
# Baseline to be used
Baseline = 'Baseline 1 MobileNetV2 ' + Dataset
# Actual Model to be used
Student = ID + ' ResNet18 ' + Dataset + ' DFT'

print('Comparing AMs for Teacher {}, Baseline {} and Student {}'.format(Teacher, Baseline, Student))

# Set Paths
TeacherPath = os.path.join(ResultsPath, Dataset, 'Teachers', Teacher)
BaselinePath = os.path.join(ResultsPath, Dataset, 'Baselines', Baseline)
StudentPath = os.path.join(ResultsPath, Dataset, Student)
AMsPath = os.path.join('Images Results', 'CAMs')
PredPath = os.path.join('Files', 'predictions AMs.pkl')
GTPath = os.path.join('Files', 'gt AMs.pkl')

# Obtain the list of images
teacher_img_list = os.listdir(os.path.join(TeacherPath, AMsPath, 'Level 0'))
teacher_img_list.sort()
baseline_img_list = os.listdir(os.path.join(BaselinePath, AMsPath, 'Level 0'))
baseline_img_list.sort()
student_img_list = os.listdir(os.path.join(StudentPath, AMsPath, 'Level 0'))
student_img_list.sort()

# Load the GT and the predictions so the comparative can be properly separed into folders
with open(os.path.join(TeacherPath, GTPath), 'rb') as f:
    GT_list = pickle.load(f)
with open(os.path.join(TeacherPath, PredPath), 'rb') as f:
    teacher_pred_list = pickle.load(f)
with open(os.path.join(BaselinePath, PredPath), 'rb') as f:
    baseline_pred_list = pickle.load(f)
with open(os.path.join(StudentPath, PredPath), 'rb') as f:
    student_pred_list = pickle.load(f)

# Create Saving Path
SavingPath = ('Comparativas/Comparativa AM {} {}').format(Teacher, Student)
SavingPathCorrections = os.path.join(SavingPath, 'Correct')
SavingPathIncorrect = os.path.join(SavingPath, 'Incorrect')
SavingPathOther = os.path.join(SavingPath, 'Other')

if os.path.isdir(SavingPath):
    shutil.rmtree(SavingPath)

os.makedirs(SavingPath)
os.mkdir(SavingPathCorrections)
os.mkdir(SavingPathIncorrect)
os.mkdir(SavingPathOther)

c = 0
for am_teacher, am_bas, am_student in zip(teacher_img_list, baseline_img_list, student_img_list):

    print('Saving image {} / {}'.format(c, len(teacher_img_list)))

    GT = GT_list[c]
    teacher_pred = teacher_pred_list[c]
    baseline_pred = baseline_pred_list[c]
    student_pred = student_pred_list[c]

    # Level 0
    img_am_teacher_l0 = Image.open(os.path.join(TeacherPath, AMsPath, 'Level 0', am_teacher))
    img_am_baseline_l0 = Image.open(os.path.join(BaselinePath, AMsPath, 'Level 0', am_bas))
    img_am_student_l0 = Image.open(os.path.join(StudentPath, AMsPath, 'Level 0', am_student))

    # Level 1
    img_am_teacher_l1 = Image.open(os.path.join(TeacherPath, AMsPath, 'Level 1', am_teacher))
    img_am_baseline_l1 = Image.open(os.path.join(BaselinePath, AMsPath, 'Level 1', am_bas))
    img_am_student_l1 = Image.open(os.path.join(StudentPath, AMsPath, 'Level 1', am_student))

    # Level 2
    img_am_teacher_l2 = Image.open(os.path.join(TeacherPath, AMsPath, 'Level 2', am_teacher))
    img_am_baseline_l2 = Image.open(os.path.join(BaselinePath, AMsPath, 'Level 2', am_bas))
    img_am_student_l2 = Image.open(os.path.join(StudentPath, AMsPath, 'Level 2', am_student))

    # Level 3
    img_am_teacher_l3 = Image.open(os.path.join(TeacherPath, AMsPath, 'Level 3', am_teacher))
    img_am_baseline_l3 = Image.open(os.path.join(BaselinePath, AMsPath, 'Level 3', am_bas))
    img_am_student_l3 = Image.open(os.path.join(StudentPath, AMsPath, 'Level 3', am_student))

    plt.figure(1, figsize=(15, 10))
    plt.subplot(4, 3, 1)
    plt.imshow(img_am_teacher_l0)
    plt.axis('off')
    plt.title('Teacher L0')

    plt.subplot(4, 3, 2)
    plt.imshow(img_am_baseline_l0)
    plt.axis('off')
    plt.title('Baseline L0')

    plt.subplot(4, 3, 3)
    plt.imshow(img_am_student_l0)
    plt.axis('off')
    plt.title('Student L0')

    plt.subplot(4, 3, 4)
    plt.imshow(img_am_teacher_l1)
    plt.axis('off')
    plt.title('Teacher L1')

    plt.subplot(4, 3, 5)
    plt.imshow(img_am_baseline_l1)
    plt.axis('off')
    plt.title('Baseline L1')

    plt.subplot(4, 3, 6)
    plt.imshow(img_am_student_l1)
    plt.axis('off')
    plt.title('Student L1')

    plt.subplot(4, 3, 7)
    plt.imshow(img_am_teacher_l2)
    plt.axis('off')
    plt.title('Teacher L2')

    plt.subplot(4, 3, 8)
    plt.imshow(img_am_baseline_l2)
    plt.axis('off')
    plt.title('Baseline L2')

    plt.subplot(4, 3, 9)
    plt.imshow(img_am_student_l2)
    plt.axis('off')
    plt.title('Student L2')

    plt.subplot(4, 3, 10)
    plt.imshow(img_am_teacher_l3)
    plt.axis('off')
    plt.title('Teacher L3')

    plt.subplot(4, 3, 11)
    plt.imshow(img_am_baseline_l3)
    plt.axis('off')
    plt.title('Baseline L3')

    plt.subplot(4, 3, 12)
    plt.imshow(img_am_student_l3)
    plt.axis('off')
    plt.title('Student L3')

    if (teacher_pred == GT) and (baseline_pred != GT) and (student_pred == GT):
        plt.savefig(os.path.join(SavingPathCorrections, '{}.png'.format(str(c).zfill(3))), dpi=200)
    elif (teacher_pred == GT) and (baseline_pred == GT) and (student_pred != GT):
        plt.savefig(os.path.join(SavingPathIncorrect, '{}.png'.format(str(c).zfill(3))), dpi=200)
    else:
        plt.savefig(os.path.join(SavingPathOther, '{}.png'.format(str(c).zfill(3))), dpi=200)

    plt.close()

    c += 1
