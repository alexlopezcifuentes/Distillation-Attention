# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x evaluateCNNs.py
chmod +x extractAMs.py
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

python extractAMs.py --Model "CIFAR100/Teachers/ResNet56C CIFAR100"
python extractAMs.py --Model "CIFAR100/Baselines/ResNet20C CIFAR100"
python extractAMs.py --Model "CIFAR100/T56C S20C/ID 5 ResNet20C CIFAR100 DFT"

#python evaluateCNNs.py --Model "CIFAR100/Teachers/ResNet32x4C CIFAR100"

#python evaluateCNNs.py --Model "CIFAR100/ID 3 ResNet20C CIFAR100 DFT"
#python evaluateCNNs.py --Model "CIFAR100/ID 4 ResNet20C CIFAR100 DFT"
#python evaluateCNNs.py --Model "CIFAR100/ResNet20C CIFAR100 AT"
#python evaluateCNNs.py --Model "CIFAR100/ID 7 ResNet20C CIFAR100 DFT"
#python evaluateCNNs.py --Model "CIFAR100/ID 3 ResNet20C CIFAR100 AT"
#python evaluateCNNs.py --Model "CIFAR100/ID 4 ResNet20C CIFAR100 AT"



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







