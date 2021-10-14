# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Train.sh

#!/bin/bash

cd ..
chmod +x trainCNNs.py

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                                                                        #
#                                                                          TRAINING SCRIPTS                                                                              #
#                                                                                                                                                                        #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Parameters
# ALPHA -> Weight for Distill Losses
# BETA -> Weight for regular CE
# DELTA -> Weight for original KD
# Example script are given for the ADE20K. Changing the dataset parameter is enough to train models for the other datasets.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              TEACHERS                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

python trainCNNs.py --Architecture ResNet50  --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet50 Teacher training'
python trainCNNs.py --Architecture ResNet152 --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet152 Teacher training'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              BASELINES                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet18 Baseline training'
python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet34 Baseline training'
python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation None --Options COMMENTS='MobileNetV2 Baseline training'


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                         KNOWLEDGE DISTILLATION                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Knowledge Distillation Method
python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18'
python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet152 ADE20K' COMMENTS='Proposed DCT approach T:R152 S:R34'
python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:MN2'

# Original Hinton KD Method. Is obtained with Alpha=1, Beta=0.1 and Delta=0
python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation KD --Options TEACHER='Teacher ResNet50 ADE20K'  BETA=0.1 COMMENTS='Original KD T:R50 S:R18'
python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation KD --Options TEACHER='Teacher ResNet152 ADE20K' BETA=0.1 COMMENTS='Original KD T:R152 S:R34'
python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation KD --Options TEACHER='Teacher ResNet50 ADE20K'  BETA=0.1 COMMENTS='Original KD T:R50 S:MN2'

# Knowledge Distillation Method + Original Hinton KD Method. It is obtained with Alpha=1, Beta=0.1 and Delta=1
python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='Proposed DFT+KD approach T:R50 S:R18'
python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet152 ADE20K' DELTA=1 BETA=0.1 COMMENTS='Proposed DFT+KD approach T:R152 S:R34'
python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='Proposed DFT+KD approach T:R50 S:MN2'


