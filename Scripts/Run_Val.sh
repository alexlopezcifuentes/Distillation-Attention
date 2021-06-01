# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x evaluateCNNs.py
chmod +x ExtractCAMs.py
chmod +x visualizeAMs.py

#python evaluateCNNs.py --Model "ADE20K/Baselines/Baseline 1 MobileNetV2 ADE20K"
#python evaluateCNNs.py --Model "ADE20K/Baselines/Baseline 1 VGG11 ADE20K"


##declare -a Models=("46" "44" "45")
#declare -a Models=("47" "48")
#
#for m in "${Models[@]}"
#  do
#    	python evaluateCNNs.py --Model "${m} ResNet18 ADE20K"
#      python extractAMs.py --Model "${m} ResNet18 ADE20K"
##      python visualizeAMs.py --Model $m
#  done
#
#declare -a Models=("1" "2")
#
#for m in "${Models[@]}"
#  do
#    	python evaluateCNNs.py --Model "${m} ResNet18 MIT67"
#      python extractAMs.py --Model "${m} ResNet18 MIT67"
#      python evaluateCNNs.py --Model "${m} ResNet18 SUN397"
#      python extractAMs.py --Model "${m} ResNet18 SUN397"
##      python visualizeAMs.py --Model $m
#  done

#python evaluateCNNs.py --Model "MIT67/Baseline 1 ResNet34 MIT67"
#python evaluateCNNs.py --Model "SUN397/Baseline 1 ResNet34 SUN397"
#python evaluateCNNs.py --Model "ADE20K/Baseline 1 ResNet34 ADE20K"



#python evaluateCNNs.py --Model "SUN397/1 ResNet34 SUN397 CRD+KD"
#python evaluateCNNs.py --Model "MIT67/1 ResNet34 MIT67 KD"


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



#python evaluateCNNs.py --Model "MIT67/2 MobileNetV2 MIT67 DFT"
#python evaluateCNNs.py --Model "MIT67/3 MobileNetV2 MIT67 DFT"
#python evaluateCNNs.py --Model "MIT67/4 MobileNetV2 MIT67 DFT"
#python evaluateCNNs.py --Model "MIT67/1 MobileNetV2 MIT67 CRD"
#python evaluateCNNs.py --Model "ADE20K/1 MobileNetV2 ADE20K CKD+KD"
python evaluateCNNs.py --Model "ADE20K/1 ResNet18 ADE20K CKD+KD"
#python evaluateCNNs.py --Model "MIT67/Baselines/Baseline 1 MobileNetV2 MIT67"
#python evaluateCNNs.py --Model "SUN397/Baselines/Baseline 1 MobileNetV2 SUN397"





