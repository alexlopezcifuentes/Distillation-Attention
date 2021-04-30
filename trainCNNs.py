import sys
sys.path.insert(0, './Libs')
sys.path.insert(0, './Libs/Datasets')
import argparse
import os
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
from getConfiguration import getConfiguration, getValidationConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
import resnet


"""
Regular CNN training code. Code for testing Distillation by Attention.

Fully developed by Alejandro Lopez-Cifuentes.
"""

# Definition of arguments. All of them are optional. If no configurations are provided the selected in
# Config/config_default.yaml will be used.

parser = argparse.ArgumentParser(description='Video Classification')
parser.add_argument('--Dataset', metavar='DIR', help='Dataset to be used', required=False)
parser.add_argument('--Architecture', metavar='DIR', help='Architecture to be used', required=False)
parser.add_argument('--Training', metavar='DIR', help='Training to be used', required=False)
parser.add_argument('--Options', metavar='DIR', nargs='+', help='an integer for the accumulator')


def train(train_loader, model, optimizer, loss_function, teacher_model=None):

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

    # Switch to train mode
    model.train()

    # Extract batch size
    batch_size = train_loader.batch_size

    # Distillation
    loss_distillation = torch.tensor(0).float().cuda()
    alpha = float(CONFIG['TRAINING']['DISTILLATION']['ALPHA'])

    for i, (mini_batch) in enumerate(train_loader):
        # Start batch_time
        start_time = time.time()
        if USE_CUDA:
            images = mini_batch['Images'].cuda()
            labels = mini_batch['Labels'].cuda()

        # Distillation Loss
        if Distillation_flag:
            # Forward through teacher
            with torch.no_grad():
                predictions_teacher, features_teacher = teacher_model(images)
                predictions_teacher = torch.argmax(predictions_teacher, dim=1)
                predictions_teacher = (predictions_teacher == labels).float()

        # CNN Forward
        output, features_student = model(images)

        # Classification Loss
        loss_class = loss_function(output, labels.long())

        # Distillation Loss
        if Distillation_flag:
            loss_distillation = Distillation.distillationLoss(CONFIG['TRAINING']['DISTILLATION']['D_LOSS'], features_student, features_teacher)

            # Supress those distillation losses that the teacher has failed
            if CONFIG['TRAINING']['DISTILLATION']['PRED_GUIDE']:
                loss_distillation *= predictions_teacher

            # Weight loss
            loss_distillation *= alpha

        # Final loss
        loss_class = loss_class.mean()
        loss_distillation = loss_distillation.mean()
        loss = loss_class + loss_distillation

        # Compute and save accuracy
        acc = utils.accuracy(output, labels, topk=(1,))

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


def validate(val_loader, model, loss_function, teacher_model=None):

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

    # Extract batch size
    batch_size = val_loader.batch_size

    # Distillation
    loss_distillation = torch.tensor(0).float()
    alpha = float(CONFIG['TRAINING']['DISTILLATION']['ALPHA'])

    with torch.no_grad():
        for i, (mini_batch) in enumerate(val_loader):
            # Start batch_time
            start_time = time.time()
            if USE_CUDA:
                images = mini_batch['Images'].cuda()
                labels = mini_batch['Labels'].cuda()

            # Distillation Loss
            if Distillation_flag:
                # Forward through teacher
                with torch.no_grad():
                    predictions_teacher, features_teacher = teacher_model(images)
                    predictions_teacher = torch.argmax(predictions_teacher, dim=1)
                    predictions_teacher = (predictions_teacher == labels).float()

            # CNN Forward
            output, features_student = model(images)

            # Classification Loss
            loss_class = loss_function(output, labels.long())

            # Distillation Loss
            if Distillation_flag:
                loss_distillation = Distillation.distillationLoss(CONFIG['TRAINING']['DISTILLATION']['D_LOSS'], features_student, features_teacher)

                # Supress those distillation losses that the teacher has failed
                if CONFIG['TRAINING']['DISTILLATION']['PRED_GUIDE']:
                    loss_distillation *= predictions_teacher

                # Weight loss
                loss_distillation *= alpha

            # Final loss
            loss_class = loss_class.mean()
            loss_distillation = loss_distillation.mean()
            loss = loss_class + loss_distillation

            # Compute and save accuracy
            acc = utils.accuracy(output, labels, topk=(1,))

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
CONFIG, dataset_CONFIG, architecture_CONFIG, training_CONFIG = getConfiguration(args)

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
ResultsPath = os.path.join(CONFIG['MODEL']['OUTPUT_DIR'], Date + ' ' + CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'])

os.mkdir(ResultsPath)
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

# ----------------------------- #
#           Networks            #
# ----------------------------- #

# Given the configuration file build the desired CNN network
if CONFIG['MODEL']['ARCH'].lower() == 'resnet18':
    model = resnet.resnet18(pretrained=CONFIG['MODEL']['PRETRAINED'],
                            num_classes=CONFIG['DATASET']['N_CLASSES'],
                            multiscale=CONFIG['TRAINING']['DISTILLATION']['MULTISCALE'])
elif CONFIG['MODEL']['ARCH'].lower() == 'resnet50':
    model = resnet.resnet50(pretrained=CONFIG['MODEL']['PRETRAINED'],
                            num_classes=CONFIG['DATASET']['N_CLASSES'],
                            multiscale=CONFIG['TRAINING']['DISTILLATION']['MULTISCALE'])
elif CONFIG['MODEL']['ARCH'].lower() == 'resnet152':
    model = resnet.resnet152(pretrained=CONFIG['MODEL']['PRETRAINED'],
                             num_classes=CONFIG['DATASET']['N_CLASSES'],
                             multiscale=CONFIG['TRAINING']['DISTILLATION']['MULTISCALE'])
else:
    print('CNN Architecture specified does not exit.')
    exit()

# Extract model parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])

if USE_CUDA:
    model.cuda()

if CONFIG['TRAINING']['DISTILLATION']['TEACHER'] != 'None':
    Distillation_flag = True
    print('Defining teacher as model from {}'.format(CONFIG['TRAINING']['DISTILLATION']['TEACHER']))

    model_teacher = Distillation.defineTeacher(CONFIG['TRAINING']['DISTILLATION']['TEACHER'],
                                               multiscale=CONFIG['TRAINING']['DISTILLATION']['MULTISCALE'])

    model_teacher_parameters = filter(lambda p: p.requires_grad, model_teacher.parameters())
    model_teacher_parameters = sum([np.prod(p.size()) for p in model_teacher_parameters])

    model_teacher.eval()
    if USE_CUDA:
        model_teacher.cuda()
else:
    Distillation_flag = False
    model_teacher = None


# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

if CONFIG['DATASET']['NAME'] == 'ADE20K' or CONFIG['DATASET']['NAME'] == 'MIT67' or CONFIG['DATASET']['NAME'] == 'SUN397':
    trainDataset = SceneRecognitionDataset(CONFIG, set='Train', mode='Train')
    valDataset = SceneRecognitionDataset(CONFIG, set='Val', mode='Val')
else:
    print('Dataset specified does not exit.')
    exit()

train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TRAIN']), shuffle=True,
                                           num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['TRAINING']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=8, pin_memory=True)

dataset_nclasses = trainDataset.nclasses

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
    print('Using Teacher Distillation training. Number of params of the teacher: {}'.format(model_teacher_parameters))
    if bool(CONFIG['TRAINING']['DISTILLATION']['MULTISCALE']):
        print('Using Multiscale activation maps distillation')
    else:
        print('Using single scale activation maps distillation')
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('----------------------------------------------------------------')

# print(model)
# GeneralPlottingUtils.saveBatchExample(train_loader, os.path.join(ResultsPath, 'Images', 'Dataset', 'Training Batch Sample.png'))
# GeneralPlottingUtils.saveBatchExample(val_loader, os.path.join(ResultsPath, 'Images', 'Dataset', 'Validation Batch Sample.png'))

# print('-' * 65)
# print('Generating histogram of samples...')

# GeneralPlottingUtils.plotDatasetHistograms(trainDataset.noun_classes, trainDataset.HistNounClasses,
#                                     os.path.join(ResultsPath, 'Images', 'Dataset'), set='Training', classtype='Noun')
# GeneralPlottingUtils.plotDatasetHistograms(trainDataset.verb_classes, trainDataset.HistVerbClasses,
#                                     os.path.join(ResultsPath, 'Images', 'Dataset'), set='Training', classtype='Verb')

# ----------------------------- #
#        Hyper Parameters       #
# ----------------------------- #

# Optimizers
if CONFIG['TRAINING']['OPTIMIZER']['NAME'] == 'SGD':
    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=float(CONFIG['TRAINING']['OPTIMIZER']['LR']), momentum=0.9, weight_decay=1e-04)
else:
    raise Exception('Optimizer {} was indicate in configuration file. This optimizer is not supported.\n'
                    'The following optimizers are supported: SGD'
                    .format(CONFIG['TRAINING']['OPTIMIZER']['NAME']))

# Scheduler
if CONFIG['TRAINING']['SCHEDULER']['NAME'] == 'STEP':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG['TRAINING']['OPTIMIZER']['LR_DECAY']),
                                                gamma=CONFIG['TRAINING']['OPTIMIZER']['GAMMA'])
else:
    raise Exception('Scheduler {} was indicate in configuration file. This Scheduler is not supported.\n'
                    'The following optimizers are supported: WARM-UP, STEP'
                    .format(CONFIG['TRAINING']['SCHEDULER']['NAME']))

# Loss Functions
if CONFIG['TRAINING']['LOSS']['NAME'] == 'CROSS ENTROPY':
    loss_function = nn.CrossEntropyLoss(reduction='none')
else:
    raise Exception('Loss function {} was indicate in {} file. This Scheduler is not supported.\n'
                    'The following optimizers are supported: Cross-Entropy'
                    .format(CONFIG['TRAINING']['LOSS']['NAME'], args.ConfigPath))


# ----------------------------- #
#           Training            #
# ----------------------------- #

# Training epochs
train_epochs = CONFIG['TRAINING']['EPOCHS']
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
    train_losses, train_accuracies = train(train_loader, model, optimizer, loss_function, model_teacher)

    # Validate one epoch
    val_losses, val_accuracies = validate(val_loader, model, loss_function, model_teacher)

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
