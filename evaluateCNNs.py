import sys
sys.path.insert(0, './Libs')
sys.path.insert(0, './Libs/Datasets')
sys.path.insert(0, './Distillation Zoo')
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import yaml
import Utils as utils
from getConfiguration import getValidationConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
import matplotlib.pyplot as plt
import resnet
import resnetCIFAR
import mobilenetv2
import pickle
import Distillation
import DFTOurs
import pytorch_ssim
import reviewKD


from DFTOurs import DFTOurs as dct


"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

evaluateCNNs.py
Python file to evaluate the models. It has to be fed with the Model path.

Fully developed by Anonymous Code Author.
"""

# Definition of arguments. All of them are optional. If no configurations are provided the selected in
# Config/config_default.yaml will be used.

parser = argparse.ArgumentParser(description='Video Classification')
parser.add_argument('--Model', metavar='DIR', help='Folder to be evaluated', required=True)


def validate(val_loader, model, teacher_model):
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
        'top1': utils.AverageMeter(),
        'top5': utils.AverageMeter(),
    }

    # Instantiate SSIM metric
    SSIM = list()
    if CONFIG['DATASET']['NAME'].lower() == 'cifar100':
        SSIM.extend((utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()))
    else:
        SSIM.extend((utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()))

    pred_list = []
    GT_list = []

    # Switch to eval mode
    model.eval()

    # Extract batch size
    batch_size = val_loader.batch_size

    # Loss Distillation
    loss_distillation = torch.tensor(0).float()

    # Auxiliar class to obtain AMs (is where the function is)
    DFT = DFTOurs.DFTOurs

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

            # CNN Forward
            output, features_student = model(images)

            if teacher_model is not None:
                output_teacher, features_teacher = teacher_model(images)

                # Compute difference between student AMs and teacher AMs
                features_student = features_student[:-1]
                features_teacher = features_teacher[:-1]

                predictions_teacher = torch.argmax(output_teacher, dim=1)
                predictions_teacher = (predictions_teacher == labels).float()
                predictions_teacher = predictions_teacher == 1

                dct_loss = dct()
                dct_loss(features_student, features_teacher)

                n_scales = len(features_teacher)
                for scale in range(n_scales):
                    AMs_student = DFT.returnAM(DFT, features_student[scale])
                    AMs_teacher = DFT.returnAM(DFT, features_teacher[scale])

                    AMs_student = AMs_student[predictions_teacher]
                    AMs_teacher = AMs_teacher[predictions_teacher]

                    ssim_loss = pytorch_ssim.ssim(torch.unsqueeze(AMs_student, dim=1), torch.unsqueeze(AMs_teacher, dim=1))
                    SSIM[scale].update(ssim_loss.item(), batch_size)

            # Classification Loss
            loss_class = loss_function(output, labels.long())

            # Final loss
            loss = loss_class + loss_distillation

            # Compute and save accuracy
            acc = utils.accuracy(output, labels, topk=(1,5))

            # Save Losses and Accuracies
            losses['total'].update(loss.item(), batch_size)
            losses['classification'].update(loss_class.item(), batch_size)
            losses['distillation'].update(loss_distillation.item(), batch_size)

            accuracies['top1'].update(acc[0].item(), batch_size)
            accuracies['top5'].update(acc[1].item(), batch_size)

            # SSIM.update(ssim_loss.item(), batch_size)

            batch_time.update(time.time() - start_time)

            # Save predictions
            pred = torch.argmax(output, dim=1)
            pred_list.extend(pred.cpu())

            # Save Ground-Truth
            GT_list.extend(labels.cpu())

            batch_time.update(time.time() - start_time)

            if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Validation Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f})\t'.
                      format(i, len(val_loader), batch_time=batch_time, accuracy=accuracies['top1']))

        with open(os.path.join(args.Model, 'Files', 'predictions.pkl'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(pred_list, filehandle)

        with open(os.path.join(args.Model, 'Files', 'gt.pkl'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(GT_list, filehandle)

        # Convert pred_list and GT_list to numpy arrays
        pred_list = torch.stack(pred_list, 0).numpy()
        GT_list = torch.stack(GT_list, 0).numpy()

        # Confusion Matrix
        CM = confusion_matrix(GT_list, pred_list, labels=np.arange(valDataset.nclasses))

        # Class Accuracy
        accuracies['class'] = utils.classAccuracy(CM)

        print('Elapsed time for evaluation {time:.3f} seconds'.format(time=time.time() - val_time_start))

    return accuracies, CM, SSIM


# ----------------------------- #
#   Global Variables & Config   #
# ----------------------------- #

global USE_CUDA, CONFIG
USE_CUDA = torch.cuda.is_available()

args = parser.parse_args()
CONFIG = getValidationConfiguration(args.Model, ResultsPath='')

# Initialize best precision
best_prec = 0

print('Evaluation starts.')
print('-' * 65)


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

# Extract model parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])

if USE_CUDA:
    model.cuda()

# Load Model to evaluate
completePath = os.path.join(args.Model, 'Models', CONFIG['MODEL']['ARCH'] + '_' + CONFIG['DATASET']['NAME'] + '_best.pth.tar')

if os.path.isfile(completePath):
    checkpoint = torch.load(completePath)
    best_prec_val = checkpoint['best_prec_val']
    best_prec_train = checkpoint['best_prec_train']

    print('Testing Model {}'.format(completePath))
    print('Reported performance on training:')
    print('Train Accuracy: {best_prec_train:.2f}%.\n'
          'Val Accuracy:   {best_prec_val:.2f}%.\n'
          .format(best_prec_train=best_prec_train, best_prec_val=best_prec_val))

    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint['model'])

    model.eval()
else:
    exit('Model ' + completePath + ' was not found.')


# Load, if so, Teacher that was used for training the model
if CONFIG['DISTILLATION']['TEACHER'] != 'None':
    teacher_path = os.path.join(CONFIG['DATASET']['NAME'], 'Teachers', CONFIG['DISTILLATION']['TEACHER'])
else:
    teacher_path = os.path.join(CONFIG['DATASET']['NAME'], 'Teachers', 'ResNet56C ' + CONFIG['DATASET']['NAME'])

print('Defining teacher as model from {}'.format(teacher_path))

model_teacher = Distillation.defineTeacher(teacher_path)

model_teacher.eval()
if USE_CUDA:
    model_teacher.cuda()

# model_teacher = None


# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

if CONFIG['DATASET']['NAME'] in ['ADE20K', 'MIT67', 'SUN397']:
    valDataset = SceneRecognitionDataset(CONFIG, set='Val', mode='Val')
    # valDataset = SceneRecognitionDataset(CONFIG, set='Train', mode='Val')
elif CONFIG['DATASET']['NAME'] in ['CIFAR100']:
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CONFIG['DATASET']['MEAN'], CONFIG['DATASET']['STD']),
    ])

    # valDataset = datasets.CIFAR100(root=CONFIG['DATASET']['ROOT'], download=True, train=False, transform=val_transform)
    valDataset = datasets.CIFAR100(root=CONFIG['DATASET']['ROOT'], download=True, train=True, transform=val_transform)

    valDataset.nclasses = 100
else:
    print('Dataset specified does not exit.')
    exit()

# val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=False,
#                                          num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=8, pin_memory=True)


dataset_nclasses = valDataset.nclasses


# ----------------------------- #
#          Information          #
# ----------------------------- #

print('Dataset loaded:')
print('Validation set. Size {} video sequences. Batch size {}. Nbatches {}'.format(len(val_loader) * int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']),
                                                                                   int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']), len(val_loader)))
print('Number of classes: {}.' .format(CONFIG['DATASET']['N_CLASSES']))
print('-' * 65)
print('Number of params: {}'. format(model_parameters))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)

# Loss Functions
if CONFIG['TRAINING']['LOSS']['NAME'] == 'CROSS ENTROPY':
    loss_function = nn.CrossEntropyLoss()
else:
    raise Exception('Loss function {} was indicate in {} file. This Scheduler is not supported.\n'
                    'The following optimizers are supported: Cross-Entropy'
                    .format(CONFIG['TRAINING']['LOSS']['NAME'], args.ConfigPath))

# ----------------------------- #
#         Evaluation            #
# ----------------------------- #

# Validate one epoch
accuracies, CM, SSIM = validate(val_loader, model, model_teacher)

print('-' * 65)
print('RESULTS FOR VALIDATION')
print('Top@1 Accuracy: {top1:.2f}%.\n'
      'Top@5 Accuracy: {top5:.2f}%.\n'
      'MCA: {mca:.2f}%.\n'
      .format(top1=accuracies['top1'].avg, top5=accuracies['top5'].avg, mca=np.mean(accuracies['class'])*100))

ssim_avg = 0
for i in range(len(SSIM)):
    print('SSIM L{}: {ssim:.2f}.'.format(i, ssim=SSIM[i].avg))
    ssim_avg += SSIM[i].avg
ssim_avg /= len(SSIM)
print('SSIM Averaged: {ssim:.2f}.'.format(ssim=ssim_avg))

# Summary Stats
sumary_dict_file = {'VALIDATION': {'ACCURACY TOP1': str(round(accuracies['top1'].avg, 2)) + ' %',
                                   'ACCURACY TOP5': str(round(accuracies['top5'].avg, 2)) + ' %',
                                   'MCA': str(round(np.mean(accuracies['class'])*100, 2)) + ' %',
                                   'SSIM': str(round(ssim_avg, 3)),
                                   },
                    'EPOCH': checkpoint['epoch'],
                    }

# Save new summary stats for the new best model
with open(os.path.join(args.Model, CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' Validation Report.yaml'), 'w') as file:
    yaml.safe_dump(sumary_dict_file, file)

print('-' * 65)
print('Saving Confunsion Matrices')

# Confusion Matrix
plt.figure()
plt.imshow(CM)
plt.ylabel('Classes', fontsize=10), plt.xlabel('Classes', fontsize=10)
plt.xticks(np.arange(len(valDataset.classes)), valDataset.classes, rotation=90, fontsize=2)
plt.yticks(np.arange(len(valDataset.classes)), valDataset.classes, rotation=0, fontsize=2)
plt.title('Confusion Matrix', fontsize=10)
plt.savefig(os.path.join(args.Model, 'Images', 'Confusion Matrix.pdf'), bbox_inches='tight', dpi=600)
plt.close()

# Accuracy per Class
plt.figure()
plt.bar(np.arange(len(accuracies['class'])), accuracies['class'], width=1, edgecolor='k')
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Classes', fontsize=15)
plt.xticks(np.arange(len(accuracies['class'])), valDataset.classes, rotation=90, fontsize=1)
plt.title('Accuracy per noun class')
plt.savefig(os.path.join(args.Model, 'Images', 'Acc Class.pdf'), bbox_inches='tight', dpi=300)
plt.close()

print('Evaluation finished!')
