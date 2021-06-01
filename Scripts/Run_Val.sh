# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x evaluateCNNs.py
chmod +x ExtractCAMs.py
chmod +x visualizeAMs.py

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                                                                        #
#                                                                         EVALUATION SCRIPTS                                                                             #
#                                                                                                                                                                        #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              TEACHERS                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

python evaluateCNNs.py --Model "ADE20K/Teachers/Teacher 1 ResNet50 ADE20K"
python evaluateCNNs.py --Model "ADE20K/Teachers/Teacher 1 ResNet152 ADE20K"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                              BASELINES                                                                                 #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

python evaluateCNNs.py --Model "ADE20K/Baselines/Baseline 1 ResNet18 ADE20K"
python evaluateCNNs.py --Model "ADE20K/Baselines/Baseline 1 ResNet32 ADE20K"
python evaluateCNNs.py --Model "ADE20K/Baselines/Baseline 1 MobileNetV2 ADE20K"

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
#                                                                         KNOWLEDGE DISTILLATION                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

python evaluateCNNs.py --Model "ADE20K/1 ResNet18 ADE20K DFT"
python evaluateCNNs.py --Model "ADE20K/1 ResNet32 ADE20K DFT"
python evaluateCNNs.py --Model "ADE20K/1 MobileNetV2 ADE20K DFT"

python evaluateCNNs.py --Model "ADE20K/1 ResNet18 ADE20K KD"
python evaluateCNNs.py --Model "ADE20K/1 ResNet32 ADE20K KD"
python evaluateCNNs.py --Model "ADE20K/1 MobileNetV2 ADE20K KD"

python evaluateCNNs.py --Model "ADE20K/1 ResNet18 ADE20K DFT+KD"
python evaluateCNNs.py --Model "ADE20K/1 ResNet32 ADE20K DFT+KD"
python evaluateCNNs.py --Model "ADE20K/1 MobileNetV2 ADE20K DFT+KD"




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







