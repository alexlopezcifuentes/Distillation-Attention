#!/bin/bash

# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Train.sh

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


python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18'
python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation AT  --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='AT approach T:R50 S:R18'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              TEACHERS                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Regular Teachers
#python trainCNNs.py --Architecture ResNet152  --Dataset ADE20K --Distillation None --Options BS_TRAIN=80 LR=0.01 COMMENTS='ResNet152 Teacher training'

#python trainCNNs.py --Architecture ResNet50   --Dataset MIT67 --Distillation None --Options LR=0.01 COMMENTS='ResNet50 Teacher training'
#python trainCNNs.py --Architecture ResNet152  --Dataset MIT67 --Distillation None --Options BS_TRAIN=80 LR=0.01 COMMENTS='ResNet152 Teacher training'

#python trainCNNs.py --Architecture ResNet50   --Dataset SUN397 --Distillation None --Options LR=0.01 COMMENTS='ResNet50 Teacher training'
#python trainCNNs.py --Architecture ResNet152  --Dataset SUN397 --Distillation None --Options BS_TRAIN=80 LR=0.01 COMMENTS='ResNet152 Teacher training'

# CIFAR100
#python trainCNNs.py --Architecture ResNet56C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS='ResNet56C Teacher training'
#python trainCNNs.py --Architecture ResNet56C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS='ResNet56C Teacher training'
#python trainCNNs.py --Architecture ResNet110C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS='ResNet110C Teacher training'
#python trainCNNs.py --Architecture ResNet32x4C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS='ResNet110C Teacher training'

# Transfer Learning
#python trainCNNs.py --Architecture ResNet34  --Dataset MIT67 --Distillation None --Options PRETRAINED=ImageNet FINETUNNING=True LR=0.01 COMMENTS="ResNet34 Teacher training. Finetunnning from ImageNet pretraining"



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              BASELINES                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet18 Baseline training'
#python trainCNNs.py --Architecture ResNet20 --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet20 Baseline training'
#python trainCNNs.py --Architecture ResNet56 --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet56 Baseline training'

# CIFAR100
#python trainCNNs.py --Architecture ResNet20C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS="ResNet20C Baseline training"


# Transfer Learning
#python trainCNNs.py --Architecture ResNet18    --Dataset MIT67 --Distillation None --Options --Options PRETRAINED=ImageNet FINETUNNING=True LR=0.1 COMMENTS="ResNet18 Baseline training. Finetunnning from ImageNet pretraining"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                         KNOWLEDGE DISTILLATION                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

                   # -------------------------------------------------------------------------------------------------------------------------------- #
                   #                                                             DCT-Based                                                            #
                   # -------------------------------------------------------------------------------------------------------------------------------- #

# Knowledge Distillation Method
#if [ $1 -eq 0 ];
#then
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18'
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options PRED_GUIDE=False TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18. No PRED_GUIDE'
#fi

# CIFAR100
#python trainCNNs.py --Architecture ResNet20C --Dataset CIFAR100 --Distillation DFT --Training SGD_CIFAR --Options TEACHER='ResNet110C CIFAR100' PRED_GUIDE=False COMMENTS='Proposed DCT approach T:R110C S:R20C.'
#python trainCNNs.py --Architecture ResNet32C --Dataset CIFAR100 --Distillation DFT --Training SGD_CIFAR --Options TEACHER='ResNet110C CIFAR100' PRED_GUIDE=False COMMENTS='Proposed DCT approach T:R110C S:R20C.'
#python trainCNNs.py --Architecture ResNet8x4C --Dataset CIFAR100 --Distillation DFT --Training SGD_CIFAR --Options TEACHER='ResNet32x4C CIFAR100' PRED_GUIDE=False COMMENTS='Proposed DCT approach T:R32x4C S:R20C.'


# Transfer Learning
#python trainCNNs.py --Architecture ResNet18    --Dataset MIT67 --Distillation DFT --Options PRETRAINED=ImageNet FINETUNNING=True TEACHER="Teacher FT 2 ResNet34 MIT67" COMMENTS="Proposed DCT approach T:R34 S:R18. Transfer Learning"

# Original Hinton KD Method. Is obtained with Alpha=1, Beta=0.1 and Delta=0
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation KD --Options TEACHER='Teacher ResNet50 ADE20K'  BETA=0.1 COMMENTS='Original KD T:R50 S:R18'

# Knowledge Distillation Method + Original Hinton KD Method. It is obtained with Alpha=1, Beta=0.1 and Delta=1
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='Proposed DFT+KD approach T:R50 S:R18'

                   # -------------------------------------------------------------------------------------------------------------------------------- #
                   #                                                                 AT                                                               #
                   # -------------------------------------------------------------------------------------------------------------------------------- #

# Knowledge Distillation Method
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation AT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='AT T:R50 S:R18'

# Knowledge Distillation Method + Original Hinton KD Method. It is obtained with Alpha=1, Beta=0.1 and Delta=1
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation AT --Options TEACHER='Teacher ResNet50 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='AT+KD T:R50 S:R18'


                   # -------------------------------------------------------------------------------------------------------------------------------- #
                   #                                                               REVIEW                                                             #
                   # -------------------------------------------------------------------------------------------------------------------------------- #

# Knowledge Distillation Method
#python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation REVIEW --Options TEACHER='Teacher ResNet152 ADE20K'  COMMENTS='REVIEW T:R152 S:R34'
#python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation REVIEW --Options TEACHER='Teacher ResNet50 ADE20K'   COMMENTS='REVIEW T:R50 S:MobileNetV2'

#if [ $1 -eq 0 ];
#then
  #python trainCNNs.py --Architecture ResNet18    --Dataset MIT67  --Distillation REVIEW  --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 MIT67'  COMMENTS='REVIEW T:R50 S:ResNet18'
#  python trainCNNs.py --Architecture ResNet34    --Dataset MIT67  --Distillation REVIEW  --Options BS_TRAIN=60 TEACHER='Teacher ResNet152 MIT67' COMMENTS='REVIEW T:R152 S:R34'
#  python trainCNNs.py --Architecture MobileNetV2 --Dataset MIT67  --Distillation REVIEW  --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 MIT67'  COMMENTS='REVIEW T:R50 S:MobileNetV2'
#fi

#python trainCNNs.py --Architecture ResNet18    --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 SUN397'   COMMENTS='REVIEW T:R50 S:ResNet18'
#python trainCNNs.py --Architecture ResNet34    --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet152 SUN397'  COMMENTS='REVIEW T:R152 S:R34'
#python trainCNNs.py --Architecture MobileNetV2 --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 SUN397'   COMMENTS='REVIEW T:R50 S:MobileNetV2'


# Knowledge Distillation Method + Original Hinton KD Method. It is obtained with Alpha=1, Beta=0.1 and Delta=1
#python trainCNNs.py --Architecture ResNet34    --Dataset ADE20K --Distillation REVIEW --Options TEACHER='Teacher ResNet152 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R152 S:R34'
#python trainCNNs.py --Architecture MobileNetV2 --Dataset ADE20K --Distillation REVIEW --Options TEACHER='Teacher ResNet50 ADE20K'   DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R50 S:MobileNetV2'

#if [ $1 -eq 0 ];
#then
#  python trainCNNs.py --Architecture ResNet18    --Dataset MIT67 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 MIT67'   DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R50 S:ResNet18'
#  python trainCNNs.py --Architecture ResNet34    --Dataset MIT67 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet152 MIT67'  DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R152 S:R34'
#  python trainCNNs.py --Architecture MobileNetV2 --Dataset MIT67 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 MIT67'   DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R50 S:MobileNetV2'
#fi

#if [ $1 -eq 1 ];
#then
#  python trainCNNs.py --Architecture ResNet18    --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 SUN397'   DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R50 S:ResNet18'
#  python trainCNNs.py --Architecture ResNet34    --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet152 SUN397'  DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R152 S:R34'
#  python trainCNNs.py --Architecture MobileNetV2 --Dataset SUN397 --Distillation REVIEW --Options BS_TRAIN=60 TEACHER='Teacher ResNet50 SUN397'   DELTA=1 BETA=0.1 COMMENTS='REVIEW+KD T:R50 S:MobileNetV2'
#fi

