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
from sklearn.metrics import confusion_matrix
import numpy as np
import yaml
import Utils as utils
from getConfiguration import getValidationConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
import matplotlib.pyplot as plt
import resnet
import mobilenetv2
import pickle


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


def validate(val_loader, model):
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

    pred_list = []
    GT_list = []

    # Switch to eval mode
    model.eval()

    # Extract batch size
    batch_size = val_loader.batch_size

    # Loss Distillation
    loss_distillation = torch.tensor(0).float()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(val_loader):
            # Start batch_time
            start_time = time.time()
            if USE_CUDA:
                images = mini_batch['Images'].cuda()
                labels = mini_batch['Labels'].cuda()

            # CNN Forward
            output, _ = model(images)

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

        with open(os.path.join('Results', args.Model, 'Files', 'predictions.pkl'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(pred_list, filehandle)

        with open(os.path.join('Results', args.Model, 'Files', 'gt.pkl'), 'wb') as filehandle:
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

    return accuracies, CM


# ----------------------------- #
#   Global Variables & Config   #
# ----------------------------- #

global USE_CUDA, CONFIG
USE_CUDA = torch.cuda.is_available()

args = parser.parse_args()
CONFIG = getValidationConfiguration(args.Model)

# Initialize best precision
best_prec = 0

print('Evaluation starts.')
print('-' * 65)


# ----------------------------- #
#           Networks            #
# ----------------------------- #
# Given the configuration file build the desired CNN network
if CONFIG['MODEL']['ARCH'].lower() == 'mobilenetv2':
    model = mobilenetv2.mobilenet_v2(pretrained=CONFIG['MODEL']['PRETRAINED'],
                                     num_classes=CONFIG['DATASET']['N_CLASSES'],
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
completePath = os.path.join('Results', args.Model, 'Models', CONFIG['MODEL']['ARCH'] + '_' + CONFIG['DATASET']['NAME'] + '_best.pth.tar')

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

    model.eval()
else:
    exit('Model ' + completePath + ' was not found.')

# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

if CONFIG['DATASET']['NAME'] == 'ADE20K' or CONFIG['DATASET']['NAME'] == 'MIT67' or CONFIG['DATASET']['NAME'] == 'SUN397':
    valDataset = SceneRecognitionDataset(CONFIG, set='Val', mode='Val')
else:
    print('Dataset specified does not exit.')
    exit()

val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=0, pin_memory=True)


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
accuracies, CM = validate(val_loader, model)

print('-' * 65)
print('RESULTS FOR VALIDATION')
print('Top@1 Accuracy: {top1:.2f}%.\n'
      'Top@5 Accuracy: {top5:.2f}%.\n'
      'MCA: {mca:.2f}%.'
      .format(top1=accuracies['top1'].avg, top5=accuracies['top5'].avg, mca=np.mean(accuracies['class'])*100))


# Summary Stats
sumary_dict_file = {'VALIDATION': {'ACCURACY TOP1': str(round(accuracies['top1'].avg, 2)) + ' %',
                                   'ACCURACY TOP5': str(round(accuracies['top5'].avg, 2)) + ' %',
                                   'MCA': str(round(np.mean(accuracies['class'])*100, 2)) + ' %',
                                   },
                    'EPOCH': checkpoint['epoch'],
                    }

# Save new summary stats for the new best model
with open(os.path.join('Results', args.Model, CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' Validation Report.yaml'), 'w') as file:
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
plt.savefig(os.path.join('Results', args.Model, 'Images', 'Confusion Matrix.pdf'), bbox_inches='tight', dpi=600)
plt.close()

# Accuracy per Class
plt.figure()
plt.bar(np.arange(len(accuracies['class'])), accuracies['class'], width=1, edgecolor='k')
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Classes', fontsize=15)
plt.xticks(np.arange(len(accuracies['class'])), valDataset.classes, rotation=90, fontsize=1)
plt.title('Accuracy per noun class')
plt.savefig(os.path.join('Results', args.Model, 'Images', 'Acc Class.pdf'), bbox_inches='tight', dpi=300)
plt.close()

print('Evaluation finished!')
