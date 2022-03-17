#!/bin/bash

# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

cd ..
chmod +x evaluateCNNs.py
chmod +x extractAMs.py
#cd "Results/ADE20K/SOTA Alpha Search/"
#chmod +x visualizeAMs.py

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                                                                        #
#                                                                         EVALUATION SCRIPTS                                                                             #
#                                                                                                                                                                        #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              TEACHERS                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#python evaluateCNNs.py --Model "CIFAR100/Teachers/T1 ResNet56C CIFAR100"
#python evaluateCNNs.py --Model "CIFAR100/Teachers/T2 ResNet56C CIFAR100"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              BASELINES                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#python evaluateCNNs.py --Model "CIFAR100/Baselines/B1 ResNet20C CIFAR100"
#python evaluateCNNs.py --Model "CIFAR100/Baselines/B2 ResNet20C CIFAR100"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                         KNOWLEDGE DISTILLATION                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# ADE
#python evaluateCNNs.py --Model "ADE20K/ResNet34 ADE20K Review"
#python evaluateCNNs.py --Model "ADE20K/ResNet34 ADE20K Review + KD"
#python evaluateCNNs.py --Model "ADE20K/MobileNetV2 ADE20K Review"
#python evaluateCNNs.py --Model "ADE20K/MobileNetV2 ADE20K Review + KD"

# MIT
#python evaluateCNNs.py --Model "MIT67/ResNet18 MIT67 Review"
#python evaluateCNNs.py --Model "MIT67/ResNet34 MIT67 Review"
#python evaluateCNNs.py --Model "MIT67/MobileNetV2 MIT67 Review"

# SUN397
#python evaluateCNNs.py --Model "SUN397/ResNet18 SUN397 Review"
#python evaluateCNNs.py --Model "SUN397/ResNet34 SUN397 Review"
#python evaluateCNNs.py --Model "SUN397/MobileNetV2 SUN397 Review"
#python evaluateCNNs.py --Model "SUN397/ResNet18 SUN397 Review + KD"
#python evaluateCNNs.py --Model "SUN397/ResNet34 SUN397 Review + KD"
#python evaluateCNNs.py --Model "SUN397/MobileNetV2 SUN397 Review + KD"


# Other validation stuff
#declare -a Alphas=("0.1" "0.2" "0.5" "0.8" "1" "1.5" "2" "0.3" "0.4" "0.6" "0.7" "0.9" "3" "5")
#for alpha in "${Alphas[@]}"
#  do
#    python evaluateCNNs.py --Model "ADE20K/Ablation Alfa/ResNet18 ADE20K DFT Alpha ${alpha}"
#  done


#declare -a Methods=("DFT" "AT" "VID" "PKT")
#declare -a Methods=("DFT" "AT" "VID" "CRD" "PKT")
#declare -a Methods=("CRD")
#declare -a Datasets=("SUN397" "MIT67")

#for m in "${Methods[@]}"
#  do
#    for d in "${Datasets[@]}"
#      do
#    	  python evaluateCNNs.py --Model "${d}/1 ResNet34 ${d} ${m}"
#    	done
#  done
#
#for m in "${Methods[@]}"
#  do
#    python evaluateCNNs.py --Model "SUN397/1 MobileNetV2 SUN397 ${m}"
#    python evaluateCNNs.py --Model "SUN397/1 MobileNetV2 SUN397 ${m}+KD"
#  done


for d in Results/ADE20K/to_evaluate/*/ ; do
#    echo "$d"
    python evaluateCNNs.py --Model "${d}"
done

for d in Results/MIT67/to_evaluate/*/ ; do
#    echo "$d"
    python evaluateCNNs.py --Model "${d}"
done


for d in Results/SUN397/to_evaluate/*/ ; do
#    echo "$d"
    python evaluateCNNs.py --Model "${d}"
done





