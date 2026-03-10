#!/bin/bash
mkdir -p data
cd data
# Remplacez par les liens réels de votre release GitHub
wget https://github.com/legio201516/mnist-speedrun/releases/download/v0.1.0/X_test.bin
wget https://github.com/legio201516/mnist-speedrun/releases/download/v0.1.0/X_train.bin
wget https://github.com/legio201516/mnist-speedrun/releases/download/v0.1.0/y_test.bin
wget https://github.com/legio201516/mnist-speedrun/releases/download/v0.1.0/y_train.bin
wget https://github.com/legio201516/mnist-speedrun/releases/download/v0.1.0/metadata.txt

