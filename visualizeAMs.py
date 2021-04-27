import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import shutil
import argparse

parser = argparse.ArgumentParser(description='CAMs Comparative')
parser.add_argument('--Model', metavar='DIR', help='Folder to be evaluated', required=True)

args = parser.parse_args()
ID = args.Model

ResultsPath = '/home/alc/Distillation-Attention/Results/'

Teacher = 'Teacher ResNet50 ADE20K'
Baseline = 'Baseline 3 ResNet18 ADE20K'
Student = ID + ' ResNet18 ADE20K'

print('Comparing AMs for Teacher {}, Baseline {} and Student {}'.format(Teacher, Baseline, Student))

TeacherPath = os.path.join(ResultsPath, Teacher)
BaselinePath = os.path.join(ResultsPath, Baseline)
StudentPath = os.path.join(ResultsPath, Student)
AMsPath = os.path.join('Images Results', 'CAMs')
PredPath = os.path.join('Files', 'predictions AMs.pkl')
GTPath = os.path.join('Files', 'gt AMs.pkl')

teacher_img_list = os.listdir(os.path.join(TeacherPath, AMsPath, 'Level 0'))
teacher_img_list.sort()
baseline_img_list = os.listdir(os.path.join(BaselinePath, AMsPath, 'Level 0'))
baseline_img_list.sort()
student_img_list = os.listdir(os.path.join(StudentPath, AMsPath, 'Level 0'))
student_img_list.sort()

with open(os.path.join(TeacherPath, GTPath), 'rb') as f:
    GT_list = pickle.load(f)
with open(os.path.join(TeacherPath, PredPath), 'rb') as f:
    teacher_pred_list = pickle.load(f)
with open(os.path.join(BaselinePath, PredPath), 'rb') as f:
    baseline_pred_list = pickle.load(f)
with open(os.path.join(StudentPath, PredPath), 'rb') as f:
    student_pred_list = pickle.load(f)

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
