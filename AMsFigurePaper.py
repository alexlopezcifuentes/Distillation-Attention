import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

AMsFigurePaper.py
Python file to plot Activation Maps and the Comparisson with respect to AT.
Python file used to create the figure from the paper.

Fully developed by Anonymous Code Author.
"""


ImgsPath = os.path.join('Data', 'ADEChallengeData2016', 'train')
Model1Path = os.path.join('Results', 'ADE20K', 'Baselines', 'Baseline 3 ResNet18 ADE20K')
Model2Path = os.path.join('Results', 'ADE20K', 'Teachers', 'Teacher ResNet50 ADE20K')
Ours = os.path.join('Results', 'ADE20K', '1 ResNet18 ADE20K DFT')
AT = os.path.join('Results', 'ADE20K', 'Old Results', '33 ResNet18 ADE20K')

RGBtransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)])

i_list = [7, 246, 199, 4, 223]
for i in i_list:
    img = Image.open(os.path.join(ImgsPath, 'ADE_train_' + str(i).zfill(8) + '.jpg'))

    img = RGBtransform(img)

    # Figure Intro
    plt.figure(1)
    plt.subplot(3, 5, 1)
    plt.imshow(img)
    plt.title('ResNet-18', fontsize=8)
    plt.axis('off')

    plt.subplot(3, 5, 6)
    plt.imshow(img)
    plt.axis('off')
    plt.title('ResNet-50', fontsize=8)

    plt.subplot(3, 5, 11)
    plt.imshow(img)
    plt.title('KDCT (Ours) ResNet-18', fontsize=8)
    plt.axis('off')


    for level in range(4):
        AM1 = Image.open(os.path.join(Model1Path, 'Images Results', 'CAMs', 'Level ' + str(level), str(i-1).zfill(3) + '.jpg'))
        AM2 = Image.open(os.path.join(Model2Path, 'Images Results', 'CAMs', 'Level ' + str(level), str(i-1).zfill(3) + '.jpg'))
        AM3 = Image.open(os.path.join(Ours, 'Images Results', 'CAMs', 'Level ' + str(level), str(i - 1).zfill(3) + '.jpg'))

        plt.subplot(3, 5, 2 + level)
        plt.imshow(AM1)
        plt.axis('off')
        plt.title('Level ' + str(level + 1), fontsize=8)

        plt.subplot(3, 5, 7 + level)
        plt.imshow(AM2)
        plt.axis('off')

        plt.subplot(3, 5, 12 + level)
        plt.imshow(AM3)
        plt.axis('off')

    plt.savefig(os.path.join('Figura Paper', str(i).zfill(8) + '.pdf'), dpi=300)


    # Figure Comparative
    plt.figure(2)
    plt.subplot(4, 5, 1)
    plt.imshow(img)
    plt.title('ResNet-18', fontsize=8)
    plt.axis('off')

    plt.subplot(4, 5, 6)
    plt.imshow(img)
    plt.axis('off')
    plt.title('ResNet-50', fontsize=8)

    plt.subplot(4, 5, 11)
    plt.imshow(img)
    plt.title('AT [10]', fontsize=8)
    plt.axis('off')

    plt.subplot(4, 5, 16)
    plt.imshow(img)
    plt.title('KDCT (Ours) ResNet-18', fontsize=8)
    plt.axis('off')


    for level in range(4):
        AM1 = Image.open(os.path.join(Model1Path, 'Images Results', 'CAMs', 'Level ' + str(level), str(i-1).zfill(3) + '.jpg'))
        AM2 = Image.open(os.path.join(Model2Path, 'Images Results', 'CAMs', 'Level ' + str(level), str(i-1).zfill(3) + '.jpg'))
        AM3 = Image.open(os.path.join(AT, 'Images Results', 'CAMs', 'Level ' + str(level), str(i - 1).zfill(3) + '.jpg'))
        AM4 = Image.open(os.path.join(Ours, 'Images Results', 'CAMs', 'Level ' + str(level), str(i - 1).zfill(3) + '.jpg'))

        plt.subplot(4, 5, 2 + level)
        plt.imshow(AM1)
        plt.axis('off')
        plt.title('Level ' + str(level + 1), fontsize=8)

        plt.subplot(4, 5, 7 + level)
        plt.imshow(AM2)
        plt.axis('off')

        plt.subplot(4, 5, 12 + level)
        plt.imshow(AM3)
        plt.axis('off')

        plt.subplot(4, 5, 17 + level)
        plt.imshow(AM4)
        plt.axis('off')

    plt.savefig(os.path.join('Figura Paper', '2_' + str(i).zfill(8) + '.pdf'), dpi=300)