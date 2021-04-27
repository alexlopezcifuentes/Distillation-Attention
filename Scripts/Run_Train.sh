# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x trainCNNs.py

# Train Baselines
#python trainCNNs.py --Architecture ResNet50 --Options PRETRAINED=False TEACHER=None LR='0.01' LR_DECAY=15 COMMENTS='Teacher training from scratch (no imagenet)'
#python trainCNNs.py --Architecture ResNet18 --Options PRETRAINED=False TEACHER=None LR='0.01' LR_DECAY=15 COMMENTS='Baseline training from scratch (no imagenet)'
#python trainCNNs.py --Architecture ResNet18 --Options PRETRAINED=False TEACHER=None LR='0.01' LR_DECAY=25 COMMENTS='Baseline training from scratch (no imagenet). Decay 25'
python trainCNNs.py --Architecture ResNet18 --Options PRETRAINED=False TEACHER=None LR='0.1' LR_DECAY=25 COMMENTS='Baseline training from scratch (no imagenet). LR 0.1'

# L2 Training
#python trainCNNs.py --Options D_LOSS=L2 ALPHA=1 DYNAMIC_SCALING='Ranges' LR=0.01 LR_DECAY=25 COMMENTS='Testing Dynamic Scaling with L2'
#python trainCNNs.py --Options D_LOSS=L2 ALPHA=1 DYNAMIC_SCALING='Mean' LR=0.01 LR_DECAY=25 COMMENTS='Testing Dynamic Scaling with L2'
#python trainCNNs.py --Options D_LOSS=L2 ALPHA=0.5 MULTISCALE=True COMMENTS='Multiscale L2 with with new AM alpha 0.5'
#python trainCNNs.py --Options D_LOSS=L2 ALPHA=1 MULTISCALE=True COMMENTS='Multiscale L2 with new AM alpha 1'

# NewL2 Training
#python trainCNNs.py --Options D_LOSS=NewL2 ALPHA=0.5 COMMENTS='Multiscale New L2 with alpha 0.5'

# DFT Training
python trainCNNs.py --Options LR=0.1 D_LOSS=DFT ALPHA=1 COMMENTS='DFT with alpha 1 and LR 0.1'
#python trainCNNs.py --Options LR=0.01 D_LOSS=DFT ALPHA=3 COMMENTS='DFT with alpha 3 and LR 0.01'
#python trainCNNs.py --Options LR=0.01 D_LOSS=DFT ALPHA=1 COMMENTS='DFT with alpha 1 and LR 0.01'

# L2* Training
#python trainCNNs.py --Options D_LOSS=L2Paper ALPHA=1 DYNAMIC_SCALING='Ranges' LR=0.01 LR_DECAY=25 COMMENTS='Testing Dynamic Scaling with L2*'
#python trainCNNs.py --Options D_LOSS=L2Paper ALPHA=1 DYNAMIC_SCALING='Mean' LR=0.01 LR_DECAY=25 COMMENTS='Testing Dynamic Scaling with L2*'


