#!/bin/bash

# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Train.sh

cd ..
chmod +x trainCNNs.py
ip4=$(/sbin/ip -o -4 addr list enp6s18 | awk '{print $4}' | cut -d/ -f1)

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


#python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18'
#python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation AT  --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='AT approach T:R50 S:R18'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              TEACHERS                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Regular Teachers
#python trainCNNs.py --Architecture ResNet152  --Dataset ADE20K --Distillation None --Options BS_TRAIN=80 LR=0.01 COMMENTS='ResNet152 Teacher training'

# CIFAR100
#python trainCNNs.py --Architecture ResNet56C  --Dataset CIFAR100 --Distillation None --Training SGD_CIFAR --Options COMMENTS='ResNet56C Teacher training'

# Transfer Learning
#python trainCNNs.py --Architecture ResNet34  --Dataset MIT67 --Distillation None --Options PRETRAINED=ImageNet FINETUNNING=True LR=0.01 COMMENTS="ResNet34 Teacher training. Finetunnning from ImageNet pretraining"



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              BASELINES                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation None --Options COMMENTS='ResNet18 Baseline training'

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
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18'
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options PRED_GUIDE=False TEACHER='Teacher ResNet50 ADE20K'  COMMENTS='Proposed DCT approach T:R50 S:R18. No PRED_GUIDE'


# CIFAR100
#python trainCNNs.py --Architecture ResNet20C --Dataset CIFAR100 --Distillation DFT --Training SGD_CIFAR --Options TEACHER='ResNet110C CIFAR100' PRED_GUIDE=False COMMENTS='Proposed DCT approach T:R110C S:R20C.'


# Transfer Learning
#python trainCNNs.py --Architecture ResNet18    --Dataset MIT67 --Distillation DFT --Options PRETRAINED=ImageNet FINETUNNING=True TEACHER="Teacher FT 2 ResNet34 MIT67" COMMENTS="Proposed DCT approach T:R34 S:R18. Transfer Learning"

# Original Hinton KD Method. Is obtained with Alpha=1, Beta=0.1 and Delta=0
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation KD --Options TEACHER='Teacher ResNet50 ADE20K'  BETA=0.1 COMMENTS='Original KD T:R50 S:R18'

# Knowledge Distillation Method + Original Hinton KD Method. It is obtained with Alpha=1, Beta=0.1 and Delta=1
#python trainCNNs.py --Architecture ResNet18    --Dataset ADE20K --Distillation DFT --Options TEACHER='Teacher ResNet50 ADE20K'  DELTA=1 BETA=0.1 COMMENTS='Proposed DFT+KD approach T:R50 S:R18'

                   # -------------------------------------------------------------------------------------------------------------------------------- #
                   #                                                                 SOTA                                                             #
                   # -------------------------------------------------------------------------------------------------------------------------------- #

declare -a DATASETS=("ADE20K" "SUN397" "MIT67")

if [ "$ip4" == '192.168.23.149' ]; then
DIST="CRD"

elif [ "$ip4" == '192.168.23.150' ]; then
DIST="PKT"

elif [ "$ip4" == '192.168.23.151' ]; then
DIST="REVIEW"

elif [ "$ip4" == '192.168.23.152' ]; then
DIST="VID"

elif [ "$ip4" == '192.168.23.210' ]; then
DIST="AT"

fi

for dataset in "${DATASETS[@]}"
  do
     python trainCNNs.py --Architecture "ResNet18"    --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet50 ${dataset}"  COMMENTS="${DIST} ResNet18 in ${dataset}"
     python trainCNNs.py --Architecture "ResNet34"    --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet152 ${dataset}" COMMENTS="${DIST} ResNet34 in ${dataset}"
     python trainCNNs.py --Architecture "MobileNetV2" --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet50 ${dataset}"  COMMENTS="${DIST} MobileNetV2 in ${dataset} "

     # + KD
     python trainCNNs.py --Architecture "ResNet18"    --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet50 ${dataset}"  DELTA=1 BETA=0.1 COMMENTS="${DIST} ResNet18 in ${dataset} + KD"
     python trainCNNs.py --Architecture "ResNet34"    --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet152 ${dataset}" DELTA=1 BETA=0.1 COMMENTS="${DIST} ResNet34 in ${dataset} + KD"
     python trainCNNs.py --Architecture "MobileNetV2" --Dataset ${dataset} --Distillation $DIST --Options TEACHER="Teacher ResNet50 ${dataset}"  DELTA=1 BETA=0.1 COMMENTS="${DIST} MobileNetV2 in ${dataset} + KD"
  done
