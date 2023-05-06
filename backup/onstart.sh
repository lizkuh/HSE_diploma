#!/bin/bash
# This file is run on instance start. Output in ./onstart.log
pip install jupyter
apt update
apt install htop
pip install tensorboardX
pip install matplotlib
pip install scikit-learn
pip install tabulate