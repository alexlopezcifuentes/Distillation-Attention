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


#declare -a KDMethods=("AT" "VID" "CRD" "PKT" "REVIEW")

# FALTA CRD!
if [ "$ip4" == '192.168.23.151' ]; then
  declare -a Percetange_Alpha=("1.1" "1.2" "1.3" "1.4" "1.5")
  declare -a KDMethods=("REVIEW" "VID" )

elif [ "$ip4" == '192.168.23.210' ]; then
  declare -a Percetange_Alpha=("1.1" "1.2" "1.3" "1.4" "1.5")
  declare -a KDMethods=("CRD")

elif [ "$ip4" == '192.168.23.211' ]; then
  declare -a Percetange_Alpha=("-1.5" "-1.4" "-1.3" "-1.2" "-1.1" "1.1" "1.2" "1.3" "1.4" "1.5")
  declare -a KDMethods=("PKT" "AT")
fi


for percentage in "${Percetange_Alpha[@]}"
  do
    for method in "${KDMethods[@]}"
      do
        python trainCNNs.py --Architecture ResNet18 --Dataset ADE20K --Distillation ${method} --Options TEACHER='Teacher ResNet50 ADE20K' PERCENTAGE_CHANGE_ALPHA=${percentage} COMMENTS="${method} T:R50 S:R18. Variation of alpha: ${percentage}*100 %"
    	done
  done