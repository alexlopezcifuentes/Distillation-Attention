import sys
sys.path.insert(0, './Libs')
sys.path.insert(0, './Distillation Zoo')
sys.path.insert(0, './Distillation Zoo/crd')
sys.path.insert(0, './Libs/Datasets')
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import datetime
import numpy as np
import shutil
import yaml
import Utils as utils
import PlottingUtils as GeneralPlottingUtils
import Distillation
import scipy.io
from getConfiguration import getConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
from SceneRecognitionDatasetCRD import SceneRecognitionDatasetCRD
from torchvision import datasets, transforms
import resnet
import resnetCIFAR
import mobilenetv2

import PlottingUtils as GenericPlottingUtils
import cv2
from PIL import Image

# Distill Models
from DFTOurs import DFTOurs
from AT import Attention
from KD import DistillKL
from PKT import PKT
from VID import VIDLoss
from criterion import CRDLoss
from SemCKD import SemCKDLoss
import reviewKD


"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

trainCNNs.py
Python file to train the models. It is fed with the configuration files from Config/ folder.
You must specify which Dataset, Architecture, Training, and Distillation to use. If not the defaults will
be used.

Fully developed by Anonymous Code Author.
"""

# Definition of arguments. All of them are optional. If no configurations are provided the selected in
# Config/config_default.yaml will be used.

parser = argparse.ArgumentParser(description='Video Classification')
parser.add_argument('--Dataset', metavar='DIR', help='Dataset to be used', required=False)
parser.add_argument('--Architecture', metavar='DIR', help='Architecture to be used', required=False)
parser.add_argument('--Training', metavar='DIR', help='Training to be used', required=False)
parser.add_argument('--Distillation', metavar='DIR', help='Distillation to be used', required=False)
parser.add_argument('--Options', metavar='DIR', nargs='+', help='an integer for the accumulator')


def train(train_loader, model, optimizer, criterion_list, teacher_model=None):

    # Start training time counter
    train_time_start = time.time()

    # Instantiate time metric
    batch_time = utils.AverageMeter()

    # Instantiate loss metric
    losses = {
        'total': utils.AverageMeter(),
        'classification': utils.AverageMeter(),
        'distillation': utils.AverageMeter(),
    }

    # Instantiate precision metric
    accuracies = {
        'classification': utils.AverageMeter(),
    }

    # Losses
    loss_function_distill = criterion_list[0]
    loss_function_kd = criterion_list[1]
    loss_function_class = criterion_list[2]

    # Switch to train mode
    model.train()

    # Extract batch size
    batch_size = train_loader.batch_size

    # Weight for Cross Entropy Loss
    beta = float(CONFIG['TRAINING']['LOSS']['BETA'])
    # Weight for Distillation Loss
    alpha = float(CONFIG['DISTILLATION']['ALPHA']) + (float(CONFIG['DISTILLATION']['ALPHA']) * float(CONFIG['DISTILLATION']['PERCENTAGE_CHANGE_ALPHA']))
    # Weight for Original Knowledge Distillation Loss
    delta = float(CONFIG['TRAINING']['KD']['DELTA'])

    # Distillation
    loss_distillation = torch.tensor(0).float().cuda()
    # Original Distillation
    loss_kd = torch.tensor(0).float().cuda()

    for i, (mini_batch) in enumerate(train_loader):
        # Start batch_time
        start_time = time.time()
        if USE_CUDA:
            if CONFIG['DATASET']['NAME'] in ['CIFAR100']:
                images = mini_batch[0].cuda()
                labels = mini_batch[1].cuda()
            else:
                images = mini_batch['Images'].cuda()
                labels = mini_batch['Labels'].cuda()
                index = mini_batch['Index'].cuda()
            if CONFIG['DISTILLATION']['D_LOSS'].lower() == 'crd':
                index = mini_batch['Index'].cuda()
                contrast_idx = mini_batch['Sample Idx'].cuda()
            elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'ckd' and images.shape[0] < batch_size:
                continue
            else:
                index = None
                contrast_idx = None

        # CNN Forward
        output_student, features_student = model(images)

        # Classification Loss
        loss_class = loss_function_class(output_student, labels.long())
        loss_class *= beta

        # Distillation Loss
        if Distillation_flag:
            # Forward through teacher
            with torch.no_grad():
                output_teacher, features_teacher = teacher_model(images)

            # Knowledge Distillation
            loss_distillation = Distillation.KnowledgeDistillation(CONFIG, loss_function_distill, features_student, features_teacher,
                                                                   output_student, output_teacher, labels, index, contrast_idx)
            loss_distillation *= alpha

            # Original Knowledge-Distillation Loss (Hinton)
            loss_kd = loss_function_kd(output_student, output_teacher)
            loss_kd *= delta

        # Final loss
        loss = loss_class + loss_distillation + loss_kd

        # Compute and save accuracy
        acc = utils.accuracy(output_student, labels, topk=(1,))

        # Save Losses and Accuracies
        losses['total'].update(loss.item(), batch_size)
        losses['classification'].update(loss_class.item(), batch_size)
        losses['distillation'].update(loss_distillation.item(), batch_size)

        accuracies['classification'].update(acc[0].item(), batch_size)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: float(loss))

        batch_time.update(time.time() - start_time)

        if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}) '
                  'Train Loss {loss.val:.3f} (avg: {loss.avg:.3f}) '
                  'Classification Loss {loss_c.val:.3f} (avg: {loss_c.avg:.3f}) '
                  'Distillation Loss {loss_d.val:.3f} (avg: {loss_d.avg:.3f}) '
                  'Train Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f}) '
                  '{et} < {eta} '
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses['total'], loss_c=losses['classification'],
                          loss_d=losses['distillation'], accuracy=accuracies['classification'],
                          et=str(datetime.timedelta(seconds=int(batch_time.sum))),
                          eta=str(datetime.timedelta(seconds=int(batch_time.avg * (len(train_loader) - i))))))

    print('Elapsed time for training {time:.3f} seconds'.format(time=time.time() - train_time_start))

    return losses, accuracies


def validate(val_loader, model, criterion_list, teacher_model=None):

    # Start validation time counter
    val_time_start = time.time()

    # Instantiate time metric
    batch_time = utils.AverageMeter()

    # Instantiate loss metric
    losses = {
        'total': utils.AverageMeter(),
        'classification': utils.AverageMeter(),
        'distillation': utils.AverageMeter(),
    }

    # Instantiate precision metric
    accuracies = {
        'classification': utils.AverageMeter(),
    }

    # Switch to eval mode
    model.eval()

    # Losses
    loss_function_distill = criterion_list[0]
    loss_function_kd = criterion_list[1]
    loss_function_class = criterion_list[2]

    # Weight for Cross Entropy Loss
    beta = float(CONFIG['TRAINING']['LOSS']['BETA'])
    # Weight for Distillation Loss
    alpha = float(CONFIG['DISTILLATION']['ALPHA'])
    # Weight for Original Knowledge Distillation Loss
    delta = float(CONFIG['TRAINING']['KD']['DELTA'])

    # Extract batch size
    batch_size = val_loader.batch_size

    # Distillation
    loss_distillation = torch.tensor(0).float()
    # Original Distillation
    loss_kd = torch.tensor(0).float().cuda()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(val_loader):
            # Start batch_time
            start_time = time.time()
            if USE_CUDA:
                if CONFIG['DATASET']['NAME'] in ['CIFAR100']:
                    images = mini_batch[0].cuda()
                    labels = mini_batch[1].cuda()
                else:
                    images = mini_batch['Images'].cuda()
                    labels = mini_batch['Labels'].cuda()
                    idx = mini_batch['Index'].cuda()
            if CONFIG['DISTILLATION']['D_LOSS'].lower() == 'ckd':
                loss_function_distill.eval()
                if images.shape[0] < batch_size:
                    continue

            # CNN Forward
            output_student, features_student = model(images)

            # Classification Loss
            loss_class = loss_function_class(output_student, labels.long())
            loss_class *= beta

            # Distillation Loss
            if Distillation_flag:
                # Forward through teacher
                output_teacher, features_teacher = teacher_model(images)

                # Knowledge Distillation
                loss_distillation = Distillation.KnowledgeDistillation(CONFIG, loss_function_distill, features_student, features_teacher,
                                                                       output_student, output_teacher, labels)
                loss_distillation *= alpha

                # Original Knowledge-Distillation Loss (Hinton)
                loss_kd = loss_function_kd(output_student, output_teacher)
                loss_kd *= delta

            # Final loss
            loss = loss_class + loss_distillation + loss_kd

            # -----------------------------------------------------------------------------------------------------------#
            # if i == 0:
            #     features_student = features_student[:-1]
            #     features_teacher = features_teacher[:-1]
            #
            #     # Loop over all the multi scale AMs
            #     for level, (features_s, features_t) in enumerate(zip(features_student, features_teacher)):
            #
            #         AM_s_list = GenericPlottingUtils.getActivationMap(features_s.detach(), images, normalization='None',
            #                                                      visualize=False, no_rgb=True)
            #
            #         j = 0
            #         AM_s = AM_s_list[j]
            #
            #         # for j, AM_s in enumerate(AM_s_list):
            #         saving_path = os.path.join(ResultsPath, 'CAMs', str(j),
            #                                    'Level ' + str(level))
            #         if not os.path.isdir(saving_path):
            #             os.makedirs(saving_path)
            #
            #         # Convert tensor to numpy array. Then save it as mat file
            #         AM_s = AM_s.cpu().numpy()
            #         scipy.io.savemat(os.path.join(saving_path, 'Student Epoch {}.mat'.format(str(epoch).zfill(3))), {'AM_s': AM_s})
            #
            #         # Save Images
            #         # im_to_save = cv2.cvtColor(AM_s.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #         # im = Image.fromarray(im_to_save)
            #         # im.save(os.path.join(saving_path, (str(epoch).zfill(3) + '.jpg')))
            #
            #         if epoch == 0:
            #             AM_t = GenericPlottingUtils.getActivationMap(features_t.detach(), images,
            #                                                          normalization='None',
            #                                                          visualize=False, no_rgb=True)
            #
            #             # Save Images
            #             AM_t = AM_t[j]
            #             # Convert tensor to numpy array. Then save it as mat file
            #             AM_t = AM_t.cpu().numpy()
            #             scipy.io.savemat(os.path.join(saving_path, 'Teacher Epoch {}.mat'.format(str(epoch).zfill(3))), {'AM_t': AM_t})
            #
            #             # im_to_save = cv2.cvtColor(AM_t.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #             # im = Image.fromarray(im_to_save)
            #             # im.save(os.path.join(saving_path, 'Teacher ' + (str(epoch).zfill(3) + '.jpg')))

            # -----------------------------------------------------------------------------------------------------------#

            # Compute and save accuracy
            acc = utils.accuracy(output_student, labels, topk=(1,))

            # Save Losses and Accuracies
            losses['total'].update(loss.item(), batch_size)
            losses['classification'].update(loss_class.item(), batch_size)
            losses['distillation'].update(loss_distillation.item(), batch_size)

            accuracies['classification'].update(acc[0].item(), batch_size)

            batch_time.update(time.time() - start_time)

            if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
                print('Validation Batch: [{0}][{1}/{2}] '
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}) '
                      'Validation Loss {loss.val:.3f} (avg: {loss.avg:.3f}) '
                      'Validation Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f}) '
                      '{et} < {eta} '
                      .format(epoch, i, len(val_loader), batch_time=batch_time, loss=losses['total'], accuracy=accuracies['classification'],
                              et=str(datetime.timedelta(seconds=int(batch_time.sum))),
                              eta=str(datetime.timedelta(seconds=int(batch_time.avg * (len(val_loader) - i))))))

        print('Elapsed time for evaluation {time:.3f} seconds'.format(time=time.time() - val_time_start))
        print('Validation results: Accuracy {accuracy.avg:.3f}.'
              .format(accuracy=accuracies['classification']))

    return losses, accuracies


# ----------------------------- #
#   Global Variables & Config   #
# ----------------------------- #

global USE_CUDA, CONFIG
USE_CUDA = torch.cuda.is_available()

args = parser.parse_args()
CONFIG, dataset_CONFIG, architecture_CONFIG, training_CONFIG, distillation_CONFIG = getConfiguration(args)

print('The following configuration is used for the training')
print(yaml.dump(CONFIG, allow_unicode=True, default_flow_style=False))

# Initialize best precision
best_prec = 0

print('Training starts.')
print('-' * 65)


# ----------------------------- #
#         Results Folder        #
# ----------------------------- #

# Create folders to save results
Date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(time.localtime().tm_mday).zfill(2) +\
       ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(time.localtime().tm_sec).zfill(2)

if int(CONFIG['TRAINING']['KD']['DELTA']) == 0:
    ResultsPath = os.path.join(CONFIG['MODEL']['OUTPUT_DIR'], CONFIG['DATASET']['NAME'], Date + ' ' +
                               CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' ' + CONFIG['DISTILLATION']['D_LOSS'])
else:
    ResultsPath = os.path.join(CONFIG['MODEL']['OUTPUT_DIR'], CONFIG['DATASET']['NAME'], Date + ' ' +
                               CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' ' + CONFIG['DISTILLATION']['D_LOSS'] + ' + KD')

os.makedirs(ResultsPath)
os.mkdir(os.path.join(ResultsPath, 'Images'))
os.mkdir(os.path.join(ResultsPath, 'Images', 'Dataset'))
os.mkdir(os.path.join(ResultsPath, 'Files'))
os.mkdir(os.path.join(ResultsPath, 'Models'))


# Copy training files to results folder
shutil.copyfile('trainCNNs.py', os.path.join(ResultsPath, 'trainCNNs.py'))


# Copy all the configuration files to results folder
with open(os.path.join(ResultsPath, 'config_' + args.Dataset + '.yaml'), 'w') as file:
    yaml.safe_dump(dataset_CONFIG, file)
with open(os.path.join(ResultsPath, 'config_' + args.Architecture + '.yaml'), 'w') as file:
    yaml.safe_dump(architecture_CONFIG, file)
with open(os.path.join(ResultsPath, 'config_' + args.Training + '.yaml'), 'w') as file:
    yaml.safe_dump(training_CONFIG, file)
with open(os.path.join(ResultsPath, 'config_' + args.Distillation + '.yaml'), 'w') as file:
    yaml.safe_dump(distillation_CONFIG, file)


# ----------------------------- #
#           Networks            #
# ----------------------------- #

if CONFIG['DISTILLATION']['D_LOSS'].lower() == 'review':
    model = reviewKD.build_review_kd(CONFIG)
else:
    # Given the configuration file build the desired CNN network
    if CONFIG['MODEL']['ARCH'].lower() == 'mobilenetv2':
        model = mobilenetv2.mobilenet_v2(pretrained=CONFIG['MODEL']['PRETRAINED'],
                                         num_classes=CONFIG['DATASET']['N_CLASSES'],
                                         multiscale=CONFIG['DISTILLATION']['MULTISCALE'])
    elif CONFIG['MODEL']['ARCH'].lower().find('resnet') != -1:
        if CONFIG['MODEL']['ARCH'].lower() in ['resnet20', 'resnet20c', 'resnet56', 'resnet56c', 'resnet32x4c', 'resnet8x4c', 'resnet110c']:
            if CONFIG['MODEL']['ARCH'].lower().find('c') != -1:
                net_name = CONFIG['MODEL']['ARCH'][:CONFIG['MODEL']['ARCH'].lower().find('c')]
            else:
                net_name = CONFIG['MODEL']['ARCH']
            model = resnetCIFAR.model_dict[net_name.lower()](num_classes=CONFIG['DATASET']['N_CLASSES'],
                                                             multiscale=CONFIG['DISTILLATION']['MULTISCALE'])
        else:
            model = resnet.model_dict[CONFIG['MODEL']['ARCH'].lower()](pretrained=CONFIG['MODEL']['PRETRAINED'],
                                                                       num_classes=CONFIG['DATASET']['N_CLASSES'],
                                                                       multiscale=CONFIG['DISTILLATION']['MULTISCALE'])

if CONFIG['MODEL']['FINETUNNING']:
    # Freezing
    print('-' * 65)
    print('Freezing first ResNet block in the model!')
    print('-' * 65)
    ct = 0
    for child in model.children():
        if ct <= 4:
            # print(str(child) + " is not trained")
            for param in child.parameters():
                param.requires_grad = False
        else:
            a = 1
            # print(str(child) + " is trained")
        ct += 1

    # print('-' * 65)
    # print('Freezing all BatchNormalization in the model!')
    # print('-' * 65)
    # count = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         # shutdown update in frozen mode
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False
    #     elif isinstance(m, nn.Sequential):
    #         for j in m.modules():
    #             if isinstance(j, nn.BatchNorm2d):
    #                 j.eval()
    #                 # shutdown update in frozen mode
    #                 j.weight.requires_grad = False
    #                 j.bias.requires_grad = False


dummy_input = torch.randn(2, 3, CONFIG['MODEL']['CROP'], CONFIG['MODEL']['CROP'])

if CONFIG['DISTILLATION']['MULTISCALE']:
    _, feat_s = model(dummy_input)


# Extract model parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])

if USE_CUDA:
    model.cuda()

if CONFIG['DISTILLATION']['TEACHER'] != 'None':
    Distillation_flag = True
    teacher_path = os.path.join(CONFIG['DATASET']['NAME'], 'Teachers', CONFIG['DISTILLATION']['TEACHER'])
    print('Defining teacher as model from {}'.format(teacher_path))

    model_teacher = Distillation.defineTeacher(teacher_path)

    model_teacher_parameters = filter(lambda p: p.requires_grad, model_teacher.parameters())
    model_teacher_parameters = sum([np.prod(p.size()) for p in model_teacher_parameters])

    model_teacher.eval()
    if USE_CUDA:
        model_teacher.cuda()

    _, feat_t = model_teacher(dummy_input.cuda())
else:
    Distillation_flag = False
    model_teacher = None


student_trainable_list = nn.ModuleList([])
student_trainable_list.append(model)


# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))


if CONFIG['DATASET']['NAME'] in ['ADE20K', 'MIT67', 'SUN397']:
    if CONFIG['DISTILLATION']['D_LOSS'] == 'CRD':
        trainDataset = SceneRecognitionDatasetCRD(CONFIG, set='Train', mode='Train')
    else:
        trainDataset = SceneRecognitionDataset(CONFIG, set='Train', mode='Train')
    valDataset = SceneRecognitionDataset(CONFIG, set='Val', mode='Val')
elif CONFIG['DATASET']['NAME'] in ['CIFAR100']:
    train_transform = transforms.Compose([
        transforms.RandomCrop(CONFIG['MODEL']['CROP'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG['DATASET']['MEAN'], CONFIG['DATASET']['STD']),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CONFIG['DATASET']['MEAN'], CONFIG['DATASET']['STD']),
    ])

    trainDataset = datasets.CIFAR100(root=CONFIG['DATASET']['ROOT'], download=True, train=True, transform=train_transform)
    valDataset = datasets.CIFAR100(root=CONFIG['DATASET']['ROOT'], download=True, train=False, transform=val_transform)

    trainDataset.nclasses = 100
else:
    print('Dataset specified does not exit.')
    exit()


train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TRAIN']), shuffle=True,
                                           num_workers=6, pin_memory=True)

val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=6, pin_memory=True)

dataset_nclasses = trainDataset.nclasses


# ----------------------------- #
#       Distillation Loss       #
# ----------------------------- #
criterion_list = nn.ModuleList([])

if CONFIG['DISTILLATION']['D_LOSS'].lower() == 'dft':
    loss_function_distill = DFTOurs()
    loss_parameters = 0
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'at':
    loss_function_distill = Attention()
    loss_parameters = 0
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'kd':
    loss_function_distill = DistillKL(T=CONFIG['DISTILLATION']['TEMPERATURE'])
    loss_parameters = 0
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'pkt':
    loss_function_distill = PKT()
    loss_parameters = 0
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'vid':
    # Get channel dimensions
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]
    loss_function_distill = nn.ModuleList([VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]).cuda()
    # add this as some parameters in VIDLoss need to be updated
    student_trainable_list.append(loss_function_distill)
    loss_parameters = filter(lambda p: p.requires_grad, loss_function_distill.parameters())
    loss_parameters = sum([np.prod(p.size()) for p in loss_parameters])
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'crd':
    s_dim = feat_s[-1].shape[1]
    t_dim = feat_t[-1].shape[1]
    loss_function_distill = CRDLoss(CONFIG, s_dim, t_dim, len(trainDataset)).cuda()
    student_trainable_list.append(loss_function_distill.embed_s)
    student_trainable_list.append(loss_function_distill.embed_t)
    loss_parameters = filter(lambda p: p.requires_grad, loss_function_distill.parameters())
    loss_parameters = sum([np.prod(p.size()) for p in loss_parameters])
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'ckd':
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]
    loss_function_distill = SemCKDLoss(feat_s, feat_t, int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TRAIN']), s_n, t_n).cuda()
    student_trainable_list.append(loss_function_distill.self_attention)
    loss_parameters = filter(lambda p: p.requires_grad, loss_function_distill.self_attention.parameters())
    loss_parameters = sum([np.prod(p.size()) for p in loss_parameters])
elif CONFIG['DISTILLATION']['D_LOSS'].lower() == 'review':
    loss_function_distill = reviewKD.HCL()
    loss_parameters = filter(lambda p: p.requires_grad, loss_function_distill.parameters())
    loss_parameters = sum([np.prod(p.size()) for p in loss_parameters])
else:
    loss_function_distill = None


# Knowledge distillation loss
criterion_list.append(loss_function_distill)

# KL divergence loss, original knowledge distillation from Hinton
criterion_list.append(DistillKL(T=CONFIG['TRAINING']['KD']['TEMPERATURE']))

# ----------------------------- #
#          Information          #
# ----------------------------- #

print('Dataset loaded:')
print('Train set. Size {} images. Batch size {}. Nbatches {}'.format(len(train_loader) * int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TRAIN']),
                                                                              int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TRAIN']), len(train_loader)))
print('Validation set. Size {} images. Batch size {}. Nbatches {}'.format(len(val_loader) * int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TEST']),
                                                                                   int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TEST']), len(val_loader)))
print('Number of classes: {}' .format(dataset_nclasses))
print('-' * 65)
print('Number of params: {}'.format(model_parameters))
if Distillation_flag:
    print('-' * 65)
    print('Using {} Distillation training. Teacher to use {}. Number of params of the teacher: {}'.format(CONFIG['DISTILLATION']['D_LOSS'], teacher_path,
                                                                                                          model_teacher_parameters))
    if bool(CONFIG['DISTILLATION']['MULTISCALE']):
        print('Using Multiscale activation maps distillation')
    else:
        print('Using single scale activation maps distillation')
    print('Extra number of parameters in the {} distillation loss {}'.format(CONFIG['DISTILLATION']['D_LOSS'], loss_parameters))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('----------------------------------------------------------------')


# ----------------------------- #
#        Hyper Parameters       #
# ----------------------------- #

# Optimizers
if CONFIG['TRAINING']['OPTIMIZER']['NAME'] == 'SGD':
    optimizer = torch.optim.SGD(params=student_trainable_list.parameters(), lr=float(CONFIG['TRAINING']['OPTIMIZER']['LR']),
                                momentum=0.9, weight_decay=float(CONFIG['TRAINING']['OPTIMIZER']['WEIGHT_DECAY']))
else:
    raise Exception('Optimizer {} was indicate in configuration file. This optimizer is not supported.\n'
                    'The following optimizers are supported: SGD'
                    .format(CONFIG['TRAINING']['OPTIMIZER']['NAME']))

# Scheduler
if CONFIG['TRAINING']['SCHEDULER']['NAME'] == 'STEP':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG['TRAINING']['SCHEDULER']['LR_DECAY']),
                                                gamma=CONFIG['TRAINING']['SCHEDULER']['GAMMA'])
elif CONFIG['TRAINING']['SCHEDULER']['NAME'] == 'MULTISTEP':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CONFIG['TRAINING']['SCHEDULER']['LR_DECAY'],
                                                     gamma=CONFIG['TRAINING']['SCHEDULER']['GAMMA'])
else:
    raise Exception('Scheduler {} was indicate in configuration file. This Scheduler is not supported.\n'
                    'The following optimizers are supported: WARM-UP, STEP'
                    .format(CONFIG['TRAINING']['SCHEDULER']['NAME']))

# Loss Functions
if CONFIG['TRAINING']['LOSS']['NAME'] == 'CROSS ENTROPY':
    loss_function_class = nn.CrossEntropyLoss()
else:
    raise Exception('Loss function {} was indicate in {} file. This Scheduler is not supported.\n'
                    'The following optimizers are supported: Cross-Entropy'
                    .format(CONFIG['TRAINING']['LOSS']['NAME'], args.ConfigPath))

# Regular Cross-Entropy Loss
criterion_list.append(loss_function_class)

# ----------------------------- #
#           Training            #
# ----------------------------- #

# Training epochs
train_epochs = int(CONFIG['TRAINING']['EPOCHS'])
actual_epoch = 0

# Metrics per epoch
train_losses_epoch = utils.EpochMeter(mode='loss')
val_losses_epoch = utils.EpochMeter(mode='loss')
train_accuracies_epoch = utils.EpochMeter()
val_accuracies_epoch = utils.EpochMeter()

# List to plot Learning Rate
lr_list = []

# Epoch Loop
for epoch in range(actual_epoch, train_epochs):
    # Epoch time start
    epoch_start = time.time()

    lr_list.append(optimizer.param_groups[0]['lr'])

    # Train one epoch
    train_losses, train_accuracies = train(train_loader, model, optimizer, criterion_list, model_teacher)

    # Validate one epoch
    val_losses, val_accuracies = validate(val_loader, model, criterion_list, model_teacher)

    # Scheduler step
    scheduler.step()

    # Update Epoch meters for loss
    train_losses_epoch.update(train_losses)
    val_losses_epoch.update(val_losses)

    # Update Epoch meters for accuracies
    train_accuracies_epoch.update(train_accuracies)
    val_accuracies_epoch.update(val_accuracies)

    GeneralPlottingUtils.plotTrainingResults(train_losses_epoch, val_losses_epoch, train_accuracies_epoch, val_accuracies_epoch,
                                             lr_list, ResultsPath, CONFIG)

    # Epoch time
    epoch_time = (time.time() - epoch_start) / 60

    # Save model
    is_best = val_accuracies['classification'].avg > best_prec
    best_prec = max(val_accuracies['classification'].avg, best_prec)
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'CONFIG': CONFIG,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss_train': train_losses['total'].avg,
        'best_loss_val': val_losses['total'].avg,
        'best_prec_train': train_accuracies['classification'].avg,
        'best_prec_val': val_accuracies['classification'].avg,
        'time_per_epoch': epoch_time,
        'model_parameters': model_parameters,
    }, is_best, ResultsPath, CONFIG['MODEL']['ARCH'] + '_' + CONFIG['DATASET']['NAME'])

    print('Elapsed time for epoch {}: {time:.3f} minutes'.format(epoch, time=epoch_time))
    print('Estimated time to finish training: {time}'.format(epoch, time=str(datetime.timedelta(seconds=int(epoch_time * (train_epochs - (epoch + 1)))))))
    print(' ')

print('Training completed.')
