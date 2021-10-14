import sys
sys.path.insert(0, './Libs')
sys.path.insert(0, './Libs/Datasets')
import PlottingUtils as GenericPlottingUtils
from PIL import Image
import cv2
import argparse
import os
import torch.utils.data
import numpy as np
import shutil
from torchvision import datasets, transforms
from getConfiguration import getValidationConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
import resnet
import mobilenetv2
import resnetCIFAR
import pickle

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

extractAMs.py
Python file to extract Activation Maps for the models. It has to be fed with the Model path.
Consider changing the parameter 'n_images2save' to save more or less Activation Maps.
Consider using after 'visualizeAMs.py' to compare the obtained Activation Maps.

Fully developed by Anonymous Code Author.
"""

# Definition of arguments. The model path is necessary.
USE_CUDA = torch.cuda.is_available()

# Number of AMs to save. Change this value to save more or less images.
n_images2save = 600

parser = argparse.ArgumentParser(description='CAMs Extraction')
parser.add_argument('--Model', metavar='DIR', help='Folder to be evaluated', required=True)

args = parser.parse_args()
CONFIG = getValidationConfiguration(args.Model)

# ----------------------------- #
#         Results Folder        #
# ----------------------------- #
ResultsPath = os.path.join('Results', args.Model, 'Images Results')

if os.path.isdir(ResultsPath):
    shutil.rmtree(ResultsPath)
os.mkdir(ResultsPath)

if not os.path.isdir(os.path.join(ResultsPath, 'CAMs')):
    os.mkdir(os.path.join(ResultsPath, 'CAMs'))


# ----------------------------- #
#            Dataset            #
# ----------------------------- #

if CONFIG['DATASET']['NAME'] in ['ADE20K', 'MIT67', 'SUN397']:
    # valDataset = ADE20KDataset(CONFIG, set='Val', mode='Val')
    valDataset = SceneRecognitionDataset(CONFIG, set='Train', mode='Val')
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

val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=8, pin_memory=True)
dataset_nclasses = valDataset.nclasses


# ----------------------------- #
#           Networks            #
# ----------------------------- #

# Given the configuration file build the desired CNN network
if CONFIG['MODEL']['ARCH'].lower() == 'mobilenetv2':
    model = mobilenetv2.mobilenet_v2(pretrained=CONFIG['MODEL']['PRETRAINED'],
                                     num_classes=CONFIG['DATASET']['N_CLASSES'],
                                     multiscale=CONFIG['DISTILLATION']['MULTISCALE'])
if CONFIG['MODEL']['ARCH'].lower().find('resnet') != -1:
    if CONFIG['MODEL']['ARCH'].lower().find('c') != -1:
        net_name = CONFIG['MODEL']['ARCH'][:CONFIG['MODEL']['ARCH'].lower().find('c')]
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
    model.eval()

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
#          AMs Saving           #
# ----------------------------- #

print('Saving {} AMs'.format(n_images2save))

prediction_list = list()
gt_list = list()
with torch.no_grad():
    for j, (mini_batch) in enumerate(val_loader):
        if USE_CUDA:
            if CONFIG['DATASET']['NAME'] in ['CIFAR100']:
                images = mini_batch[0].cuda()
                labels = mini_batch[1].cuda()
            else:
                images = mini_batch['Images'].cuda()
                labels = mini_batch['Labels'].cuda()

        # CNN Forward
        output, features_ms = model(images)
        features_ms = features_ms[:-1]

        # Loop over all the multi scale AMs
        for level, features in enumerate(features_ms):

            saved_images = j * int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST'])

            saving_path = os.path.join(ResultsPath, 'CAMs', 'Level ' + str(level))
            if not os.path.isdir(saving_path):
                os.mkdir(saving_path)

            AMs = GenericPlottingUtils.getActivationMap(features, images, normalization='minmax', visualize=True)

            # Save Images
            for i, AM in enumerate(AMs):
                img_rgb = (GenericPlottingUtils.tensor2numpy(images[i, :], mean=CONFIG['DATASET']['MEAN'], STD=CONFIG['DATASET']['STD']) * 255).astype(np.uint8)
                im_to_save = np.concatenate((img_rgb, cv2.cvtColor(AM.astype(np.uint8), cv2.COLOR_BGR2RGB)), axis=1)

                GT = valDataset.classes[labels[i].item()]
                pred = valDataset.classes[torch.argmax(output, dim=1)[i].item()]

                if level == (len(features_ms) - 1):
                    gt_list.append(GT)
                    prediction_list.append(pred)

                if not CONFIG['DATASET']['NAME'] in ['CIFAR100']:
                    cv2.putText(im_to_save, ("GT  : {}.".format(GT)),
                                org=(10, 20), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=[0, 0, 0])
                    cv2.putText(im_to_save, ("Pred: {}".format(pred)),
                                org=(10, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=[0, 0, 0])

                # Save Image
                im = Image.fromarray(im_to_save)
                im.save(os.path.join(saving_path, (str(saved_images).zfill(3) + '.jpg')))

                saved_images += 1

        if saved_images > n_images2save:
            break

    with open(os.path.join('Results', args.Model, 'Files', 'predictions AMs.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(prediction_list, filehandle)

    with open(os.path.join('Results', args.Model, 'Files', 'gt AMs.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(gt_list, filehandle)

