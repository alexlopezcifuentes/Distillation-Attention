#!/bin/bash

echo ========================================================================
echo "Starting to download MIT67 Dataset..."
echo ========================================================================

# Download original zip file
wget -nc -P ./Data/MITIndoor67 http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/MITIndoor67.zip

# Unzip file
unzip -n ./Data/MITIndoor67/MITIndoor67.zip -d ./Data/MITIndoor67

# Remove zip file
rm ./Data/MITIndoor67/MITIndoor67.zip

echo ========================================================================
echo "MIT67 Dataset ready to use!"
echo ========================================================================
