#!/bin/bash

echo ========================================================================
echo "Starting to download SUN397 Dataset..."
echo ========================================================================

# Download original zip file
#wget -nc -P ./Data/SUN397 http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/SUN397.zip

# Unzip file
unzip -n ./Data/SUN397/SUN397.zip -d ./Data/SUN397

# Remove zip file
rm ./Data/SUN397/SUN397.zip

echo ========================================================================
echo "SUN397 Dataset ready to use!"
echo ========================================================================

