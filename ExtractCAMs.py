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
import Utils as utils
from getConfiguration import getValidationConfiguration
from SceneRecognitionDataset import SceneRecognitionDataset
import resnet
import pickle

"""
Script to extract Class Activation Maps from EPIC Kitchens Dataset
given a trained Action Recognition Model

"""

# Definition of arguments. The model path is necessary.
USE_CUDA = torch.cuda.is_available()

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

if CONFIG['DATASET']['NAME'] == 'ADE20K' or CONFIG['DATASET']['NAME'] == 'MIT67' or CONFIG['DATASET']['NAME'] == 'SUN397':
    # valDataset = ADE20KDataset(CONFIG, set='Val', mode='Val')
    valDataset = SceneRecognitionDataset(CONFIG, set='Train', mode='Val')
else:
    print('Dataset specified does not exit.')
    exit()

val_loader = torch.utils.data.DataLoader(valDataset, batch_size=int(CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']), shuffle=False,
                                         num_workers=0, pin_memory=True)
dataset_nclasses = valDataset.nclasses


# ----------------------------- #
#           Networks            #
# ----------------------------- #

# Given the configuration file build the desired CNN network
if CONFIG['MODEL']['ARCH'].lower() == 'resnet18':
    model = resnet.resnet18(pretrained=False, num_classes=CONFIG['DATASET']['N_CLASSES'], multiscale=True)
elif CONFIG['MODEL']['ARCH'].lower() == 'resnet50':
    model = resnet.resnet50(pretrained=False, num_classes=CONFIG['DATASET']['N_CLASSES'], multiscale=True)

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


n_images2save = 600

print('Saving {} AMs'.format(n_images2save))

prediction_list = list()
gt_list = list()
with torch.no_grad():
    for j, (mini_batch) in enumerate(val_loader):
        if USE_CUDA:
            images = mini_batch['Images'].cuda()
            labels = mini_batch['Labels'].cuda()

        # CNN Forward
        output, features_ms = model(images)

        # Loop over all the multi scale AMs
        for level, features in enumerate(features_ms):

            saved_images = j * CONFIG['VALIDATION']['BATCH_SIZE']['BS_TEST']

            saving_path = os.path.join(ResultsPath, 'CAMs', 'Level ' + str(level))
            if not os.path.isdir(saving_path):
                os.mkdir(saving_path)

            AMs = GenericPlottingUtils.getActivationMap(features, images, normalization='minmax', visualize=True)

            # Save Images
            for i, AM in enumerate(AMs):
                img_rgb = (GenericPlottingUtils.tensor2numpy(images[i, :], mean=valDataset.mean, STD=valDataset.STD) * 255).astype(np.uint8)
                im_to_save = np.concatenate((img_rgb, cv2.cvtColor(AM.astype(np.uint8), cv2.COLOR_BGR2RGB)), axis=1)

                GT = valDataset.classes[labels[i].item()]
                pred = valDataset.classes[torch.argmax(output, dim=1)[i].item()]

                if level == 3:
                    gt_list.append(GT)
                    prediction_list.append(pred)

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

