# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

cd ..
chmod +x evaluateCNNs.py
chmod +x ExtractCAMs.py
chmod +x visualizeAMs.py

#python ExtractCAMs.py --Model "Teacher ResNet50 ADE20K"
#python ExtractCAMs.py --Model "Baseline 4 ResNet18 ADE20K"
#python ExtractCAMs.py --Model "11 ResNet18 ADE20K"


declare -a Models=("43" "44" "45")

for m in "${Models[@]}"
  do
#    	python evaluateCNNs.py --Model "${m} ResNet18 ADE20K"
#      python ExtractCAMs.py --Model "${m} ResNet18 ADE20K"
      python visualizeAMs.py --Model $m
  done








