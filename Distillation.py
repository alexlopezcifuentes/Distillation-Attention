import os
import torch
import torch.fft
import resnet
import mobilenetv2
from getConfiguration import getValidationConfiguration

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

Distillation.py
Python file to model the cases for the different Distillation algorithms.

Fully developed by Anonymous Code Author.
"""

def defineTeacher(teacher_model):
    # Get config from teachers folder
    CONFIG_teacher = getValidationConfiguration(teacher_model)

    # Given the configuration file build the desired CNN network
    if CONFIG_teacher['MODEL']['ARCH'].lower() == 'mobilenetv2':
        model_teacher = mobilenetv2.mobilenet_v2(num_classes=CONFIG_teacher['DATASET']['N_CLASSES'],
                                                 multiscale=CONFIG_teacher['DISTILLATION']['MULTISCALE'])
    else:
        model_teacher = resnet.model_dict[CONFIG_teacher['MODEL']['ARCH'].lower()](num_classes=CONFIG_teacher['DATASET']['N_CLASSES'],
                                                                                   multiscale=CONFIG_teacher['DISTILLATION']['MULTISCALE'])

    # Load weights
    completeTeacherPath = os.path.join('Results', teacher_model, 'Models', CONFIG_teacher['MODEL']['ARCH'] + '_' +
                                       CONFIG_teacher['DATASET']['NAME'] + '_best.pth.tar')

    checkpoint_teacher = torch.load(completeTeacherPath)
    model_teacher.load_state_dict(checkpoint_teacher['model_state_dict'])

    print('Teacher weights loaded!')

    return model_teacher


def KnowledgeDistillation(CONFIG, loss_function, features_student, features_teacher, output_student, output_teacher, labels,
                          index=None, contrast_idx=None):

    if CONFIG['DISTILLATION']['D_LOSS'].lower() == 'dft':
        # Select all features but the last one
        features_student = features_student[:-1]
        features_teacher = features_teacher[:-1]
        # Obtain loss
        loss_distillation = loss_function(features_student, features_teacher)
        if CONFIG['DISTILLATION']['PRED_GUIDE']:
            # Suppress (zero) those distillation losses from image samples that the teacher has not correctly predict
            predictions_teacher = torch.argmax(output_teacher, dim=1)
            predictions_teacher = (predictions_teacher == labels).float()
            loss_distillation *= predictions_teacher

        loss_distillation = loss_distillation.mean()

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'at':
        features_student = features_student[:-1]
        features_teacher = features_teacher[:-1]
        loss_distillation = loss_function(features_student, features_teacher)
        loss_distillation = sum(loss_distillation)

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'kd':
        loss_distillation = loss_function(output_student, output_teacher)

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'pkt':
        features_student = features_student[-1]
        features_teacher = features_teacher[-1]
        loss_distillation = loss_function(features_student, features_teacher)

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'vid':
        g_s = features_student[1:-1]
        g_t = features_teacher[1:-1]
        loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, loss_function)]
        loss_distillation = sum(loss_group)

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'crd':
        f_s = features_student[-1]
        f_t = features_teacher[-1]
        if index is not None:
            loss_distillation = loss_function(f_s, f_t, index, contrast_idx)
        else:
            loss_distillation = torch.tensor(0).float().cuda()

    elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'ckd':
        s_value, f_target, weight = loss_function.self_attention(features_student[1:-1], features_teacher[1:-1])
        loss_distillation = loss_function(s_value, f_target, weight)

    return loss_distillation
