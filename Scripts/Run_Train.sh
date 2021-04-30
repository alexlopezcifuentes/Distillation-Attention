# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x trainCNNs.py

# Train Baselines
#python trainCNNs.py --Architecture ResNet152 --Dataset MIT67 --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher training MIT67'
#python trainCNNs.py --Architecture ResNet152 --Dataset MIT67 --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher training MIT67'
#python trainCNNs.py --Architecture ResNet152 --Dataset SUN397 --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher training SUN397'

#python trainCNNs.py --Architecture ResNet50 --Dataset MIT67 --Options TEACHER=None LR='0.01' LR_DECAY=15 COMMENTS='Teacher training MIT67'
#python trainCNNs.py --Architecture ResNet50 --Dataset SUN397 --Options TEACHER=None LR='0.01' LR_DECAY=15 COMMENTS='Teacher training SUN397'

#python trainCNNs.py --Architecture ResNet18 --Options PRETRAINED=False TEACHER=None LR='0.01' LR_DECAY=15 COMMENTS='Baseline training from scratch (no imagenet)'
#python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Options PRETRAINED='ImageNet' TEACHER=None LR='0.1' LR_DECAY=25 COMMENTS='Baseline training from imagenet. Decay 25'

#python trainCNNs.py --Architecture ResNet18 --Dataset MIT67 --Options TEACHER=None LR='0.1' LR_DECAY=25 COMMENTS='Baseline training from scratch MIT67. LR 0.1'
#python trainCNNs.py --Architecture ResNet18 --Dataset SUN397 --Options TEACHER=None LR='0.1' LR_DECAY=25 COMMENTS='Baseline training from scratch SUN397. LR 0.1'

python trainCNNs.py --Architecture ResNet152 --Dataset ADE20K --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher R152 training ADE20K'
#python trainCNNs.py --Architecture ResNet152 --Dataset MIT67 --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher training MIT67'
#python trainCNNs.py --Architecture ResNet152 --Dataset SUN397 --Options TEACHER=None LR=0.01 BS_TRAIN=55 LR_DECAY=15 COMMENTS='Teacher training SUN397'


# NewL2 Training
#python trainCNNs.py --Options D_LOSS=NewL2 ALPHA=0.5 COMMENTS='Multiscale New L2 with alpha 0.5'


# DFT Training
python trainCNNs.py --Dataset ADE20K --Options PRETRAINED='ImageNet' D_LOSS=DFT ALPHA=1 COMMENTS='DFT with alpha 1 and LR 0.1. ImageNet pretraining'
python trainCNNs.py --Dataset ADE20K --Options D_LOSS=DFT ALPHA=1 COMMENTS='DFT with alpha 1 and LR 0.1'
#python trainCNNs.py --Dataset ADE20K --Options D_LOSS=DFT ALPHA=1 PRED_GUIDE=True COMMENTS='DFT with alpha 1 and LR 0.1. Using predictions from teacher to guide distill.'


python trainCNNs.py --Dataset MIT67  --Options TEACHER='Teacher ResNet50 MIT67' LR=0.1 D_LOSS=DFT ALPHA=1 COMMENTS='MIT67 DFT with alpha 1 and LR 0.1'
python trainCNNs.py --Dataset SUN397 --Options TEACHER='Teacher ResNet50 SUN397' LR=0.1 D_LOSS=DFT ALPHA=1 COMMENTS='SUN397 DFT with alpha 1 and LR 0.1'
python trainCNNs.py --Dataset MIT67  --Options TEACHER='Teacher ResNet50 MIT67' LR=0.01 D_LOSS=DFT ALPHA=1 COMMENTS='MIT67 DFT with alpha 1 and LR 0.01'
python trainCNNs.py --Dataset SUN397 --Options TEACHER='Teacher ResNet50 SUN397' LR=0.01 D_LOSS=DFT ALPHA=1 COMMENTS='SUN397 DFT with alpha 1 and LR 0.01'


