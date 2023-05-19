#!/bin/bash
# This file is run on instance start. Output in ./onstart.log
pip install jupyter
apt update
apt install htop
pip install tensorboardX
pip install tensorboard

pip install matplotlib
pip install scikit-learn
pip install tabulate

pip install datasets loralib sentencepiece
# is it okay version?
pip install transformers
# is it okay version?
pip install peft
pip install bitsandbytes==0.37.2
# pip install bitsandbytes

apt install vim -y
apt install zip -y
df -h . >> disk_stats.txt