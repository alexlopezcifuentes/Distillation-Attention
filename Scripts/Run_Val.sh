# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x evaluateCNNs.py
chmod +x ExtractCAMs.py
chmod +x visualizeAMs.py

#python evaluateCNNs.py --Model "Teacher ResNet152 SUN397"

#python evaluateCNNs.py --Model "Teacher ResNet50 SUN397"
#python ExtractCAMs.py --Model "Teacher ResNet50 SUN397"
#python evaluateCNNs.py --Model "Baseline 1 ResNet18 MIT67"
#python evaluateCNNs.py --Model "Baseline 1 ResNet18 SUN397"
#python ExtractCAMs.py --Model "Baseline 1 ResNet18 MIT67"
#python ExtractCAMs.py --Model "Baseline 1 ResNet18 SUN397"
#python evaluateCNNs.py --Model "Baseline 6 ResNet18 ADE20K"


python evaluateCNNs.py --Model "Teacher ResNet152 ADE20K"

#declare -a Models=("46" "44" "45")
declare -a Models=("47" "48")

for m in "${Models[@]}"
  do
    	python evaluateCNNs.py --Model "${m} ResNet18 ADE20K"
      python ExtractCAMs.py --Model "${m} ResNet18 ADE20K"
#      python visualizeAMs.py --Model $m
  done

declare -a Models=("1" "2")

for m in "${Models[@]}"
  do
    	python evaluateCNNs.py --Model "${m} ResNet18 MIT67"
      python ExtractCAMs.py --Model "${m} ResNet18 MIT67"
      python evaluateCNNs.py --Model "${m} ResNet18 SUN397"
      python ExtractCAMs.py --Model "${m} ResNet18 SUN397"
#      python visualizeAMs.py --Model $m
  done








