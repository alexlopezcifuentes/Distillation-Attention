#!/bin/bash

echo ========================================================================
echo "Starting to download ADE20K Dataset..."
echo ========================================================================

# Download original zip file
wget -nc -P ./Data/ADEChallengeData2016 http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# Unzip file
unzip -n ./Data/ADEChallengeData2016/ADEChallengeData2016.zip -d ./Data/ADEChallengeData2016


# Move and rename training and validation folders
mv  -v ./Data/ADEChallengeData2016/ADEChallengeData2016/images/training ./Data/ADEChallengeData2016/train
mv  -v ./Data/ADEChallengeData2016/ADEChallengeData2016/images/validation ./Data/ADEChallengeData2016/val

# Remove extra annotations and zip file.
rm -r ./Data/ADEChallengeData2016/ADEChallengeData2016/
rm ./Data/ADEChallengeData2016/ADEChallengeData2016.zip

echo ========================================================================
echo "ADE20K Dataset ready to use!"
echo ========================================================================
