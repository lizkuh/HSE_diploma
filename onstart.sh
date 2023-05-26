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

# to run evaluateFromCodeXGlue
pip install tree-sitter==0.2.0

apt install vim -y
apt install zip -y

mkdir /root/experiments/
mkdir /root/results/

df -h . >> disk_stats.txt