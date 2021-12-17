import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import pickle
import torchvision

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

PlottingUtils.py
Bunch of utils to plot things.

Fully developed by Anonymous Code Author.
"""

def plotDatasetHistograms(classes, hist, path, set = '', classtype='', save=True):
    """
    Computes the histogram of classes for the given dataloader
    :param dataloader: Pytorch dataloader to compute the histogram
    :param classes: Classes names
    :param set: Indicates the set. Training or validation
    :param ePrint: Enables the information printing
    :return: Histogram of classes
    """

    if save:
        plt.figure()
        plt.bar(np.arange(len(classes)), hist, width=1, edgecolor='k')
        plt.ylabel('Number of Frames', fontsize=15)
        plt.xlabel('Classes', fontsize=15)
        plt.xticks(np.arange(len(classes)), classes, rotation=90, fontsize=4)
        plt.title(classtype + ' Histogram of frames for ' + set)
        plt.savefig(os.path.join(path, set + ' ' + classtype + ' Histogram.pdf'), bbox_inches='tight', dpi=300)
        plt.close()


def unNormalizeImage(image, mean=[0.43216, 0.394666, 0.37645], STD=[0.22803, 0.22145, 0.216989]):
    """
    Unnormalizes a numpy array given mean and STD
    :param image: Image to unormalize
    :param mean: Mean
    :param STD: Standard Deviation
    :return: Unnormalize image
    """
    for i in range(0, image.shape[0]):
        image[i, :, :] = (image[i, :, :] * STD[i]) + mean[i]
    return image


def tensor2numpy(image, filename=None, label="Label", to_opencv=False, save=False, display=False,
                 mean= [0.43216, 0.394666, 0.37645], STD=[0.22803, 0.22145, 0.216989]):
    """
    Function to plot a PyTorch Tensor image
    :param image: Image to display in Tensor format
    :param filename: Mean of the normalization
    :param label: (Optional) Ground-truth label
    :param save:
    :param display:
    :return: None
    """

    if image.device.type == 'cuda':
        image = image.cpu()

    if len(image.shape) == 4:
        image = torch.squeeze(image)

    # Convert PyTorch Tensor to Numpy array
    npimg = image.numpy()
    # # Unnormalize image
    npimg = unNormalizeImage(npimg, mean, STD)
    # Change from (chns, rows, cols) to (rows, cols, chns)
    npimg = np.transpose(npimg, (1, 2, 0))

    # Convert to RGB if gray-scale
    if npimg.shape[2] is 1:
        rgbArray = np.zeros((npimg.shape[0], npimg.shape[1], 3), 'float32')
        rgbArray[:, :, 0] = npimg[:, :, 0]
        rgbArray[:, :, 1] = npimg[:, :, 0]
        rgbArray[:, :, 2] = npimg[:, :, 0]
        npimg = rgbArray

    # Convert from RGB to BGR
    if to_opencv:
        npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)

    # Display image
    if display:
        plt.figure()
        plt.imshow(npimg)
        plt.title(label)

    # Save Image
    if save:
        im = Image.fromarray(np.uint8(npimg * 255))
        im.save(filename)

    return npimg


def plotTrainingResults(train_losses, val_losses, train_accuracies, val_accuracies, lr_list, ResultsPath, CONFIG):

    # ----------------------------#
    #            GRAPHS           #
    # ----------------------------#
    with plt.style.context('ggplot'):
        x = np.arange(0, len(train_losses.total), 1)

        # Training and validation losses per epoch
        plt.figure()
        plt.plot(train_losses.total, 'r', label='Training')
        plt.plot(val_losses.total, 'b', label='Validation')
        plt.fill_between(x, train_losses.total_down, train_losses.total_up, alpha=0.2, color='r')
        plt.fill_between(x, val_losses.total_down, val_losses.total_up, alpha=0.2, color='b')
        plt.ylabel('Loss'), plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(ResultsPath, 'Images', 'Epoch Losses.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # Training and validation accuracies per epoch
        plt.figure()
        plt.plot(train_accuracies.classification, 'r', label='Classification Training')
        plt.plot(val_accuracies.classification, 'b', label='Classification Validation')
        plt.ylabel('Accuracy'), plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(ResultsPath, 'Images', 'Epoch Accuracies.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # LR per epoch
        plt.figure()
        plt.plot(lr_list, 'r', label='Training')
        plt.ylabel('Learning Rate'), plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(ResultsPath, 'Images', 'Epoch LR.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # Summary Images
        fig, axs = plt.subplots(4, sharex=True)
        fig.set_size_inches(6, 10)
        fig.suptitle(CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' ' + CONFIG['DISTILLATION']['D_LOSS'])
        axs[0].plot(train_losses.total, 'r', label='Training')
        axs[0].plot(val_losses.total, 'b', label='Validation')
        axs[0].fill_between(x, train_losses.total_down, train_losses.total_up, alpha=0.2, color='r')
        axs[0].fill_between(x, val_losses.total_down, val_losses.total_up, alpha=0.2, color='b')
        axs[0].set(ylabel='Loss')
        axs[0].legend()
        axs[1].plot(train_losses.classification, 'r', label='Classification Training')
        axs[1].plot(val_losses.classification, 'b', label='Classification Validation')
        if np.sum(train_losses.distill) > 0:
            axs[1].plot(train_losses.distill, 'r--', label='Distill Training')
            axs[1].plot(val_losses.distill, 'b--', label='Distill Validation')
        axs[1].set(ylabel='Loss')
        axs[1].legend()
        axs[2].plot(train_accuracies.classification, 'r', label='Training')
        axs[2].plot(val_accuracies.classification, 'b', label='Validation')
        axs[2].set(ylabel='Accuracy')
        axs[2].legend()
        axs[3].plot(lr_list, 'r', label='Training')
        axs[3].set(xlabel='Epoch', ylabel='Learning Rate')
        plt.savefig(os.path.join(ResultsPath, 'Images', CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' Summary Figures.pdf'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    # ----------------------------#
    #            FILES            #
    # ----------------------------#
    # Lists are saved with Pickle

    # Training and validation losses per epoch
    with open(os.path.join(ResultsPath, 'Files', 'train_loss_list.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_losses.total, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'val_loss_list.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(val_losses.total, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'train_loss_list_low.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_losses.total_down, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'train_loss_list_up.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_losses.total_up, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'val_loss_list_low.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(val_losses.total_down, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'val_loss_list_up.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(val_losses.total_up, filehandle)

    # Training and validation accuracies per epoch
    with open(os.path.join(ResultsPath, 'Files', 'train_accuracy_list.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_accuracies.classification, filehandle)
    with open(os.path.join(ResultsPath, 'Files', 'val_accuracy_list.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(val_accuracies.classification, filehandle)


    # LR per epoch
    with open(os.path.join(ResultsPath, 'Files', 'lr_list.pkl'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(lr_list, filehandle)


def getActivationMap(features, RGBbatch=None, visualize=True, resize=True, normalization='none', no_rgb=False):
    """
    Funtion to get the Activation Maps from a set of features and blend them with the original RGB Image
    :param feature_conv: Feature tensor
    :param RGBbatch: RGB tensor from where the features have been extracted. We only need the shape
    :param visualize: Boolean variable that enables the display
    :param resize: Resize size if RGBbatch is None.
    :param normalization: Normalizacion to be used in the Activation Maps.
    :return:
    """

    if visualize and RGBbatch is None:
        exit()
    if visualize:
        normalization = 'minmax'

    bs, rgb_chns, h, w = features.size()

    # Obtain the Activation Maps for each of the images.
    # Activation map without taking into account the class predicted
    CAMs = returnAM(features, RGBbatch, resize, normalization=normalization)

    if visualize:
        # Render all the Activation Maps with the RGB frames for visualization
        CAMs_list_display = list()
        for i in range(bs):
            # Render the activation map and the RGB image
            img = tensor2numpy(RGBbatch[i, :], to_opencv=True) * 255

            # Colormap for CAM
            # Convert to Uint8 and normalize values to 255
            CAM = CAMs[i, :].cpu().numpy()
            CAM = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)

            # Combine both images
            if no_rgb:
                CAMs_list_display.append(CAM)
            else:
                CAMs_list_display.append(CAM * 0.5 + img * 0.5)

        CAMs = CAMs_list_display

    return CAMs


def returnAM(feature_conv, RGBbatch=None, resize=True, resizeSize=None, normalization=None):
    '''
    Funtion to, given a feature tensor, compute the Activation Map.
    :param feature_conv: Feature tensor
    :param RGBbatch: RGB tensor from where the features have been extracted. We only need the shape
    :param resize: Boolean variable to resize the Activation Map or not
    :param resizeSize: Set the resize size desired, if RGBBatch is none and we want to resize.
    :param normalization: Normalization to be applied in the Activation Map
    :return:
    '''

    # Obtain sizes from the features
    bs, nc, h, w = feature_conv.shape

    # Obtain features for a frame
    cam = feature_conv.pow(2).mean(1)

    if resize:
        # Define the upsampling size.
        if RGBbatch is not None:
            _, _, h_original, w_original = RGBbatch.shape
        else:
            h_original = resizeSize[0]
            w_original = resizeSize[1]

        # Resize to create an image
        cam = torchvision.transforms.functional.resize(cam, (h_original, w_original))

        h = h_original
        w = w_original

    if normalization.lower() == 'log_softmax':
        LogSoftmax = torch.nn.LogSoftmax(dim=1)
        cam = LogSoftmax(cam.view(bs, -1))
        cam = cam.view(bs, h, w)
    elif normalization.lower() == 'softmax':
        Softmax = torch.nn.Softmax(dim=1)
        cam = Softmax(cam.view(bs, -1))
        cam = cam.view(bs, h, w)
    elif normalization.lower() == 'l2':
        cam = cam.view(bs, -1)
        cam = torch.nn.functional.normalize(cam, p=2, dim=1)
        cam = cam.view(bs, h, w)
    elif normalization.lower() == 'minmax':

        # cam = cam.view(bs, -1)
        # min = torch.min(cam, dim=1)[0]
        # max = torch.max(cam, dim=1)[0]
        # cam = cam - min[:, None]
        # cam = cam / max[:, None]
        # cam = cam.view(bs, h, w)

        min_v = torch.min(torch.min(cam, dim=1)[0], dim=1)[0][:, None, None]
        max_v = torch.max(torch.max(cam, dim=1)[0], dim=1)[0][:, None, None]
        cam = (cam - min_v) / (max_v - min_v)

    return cam
