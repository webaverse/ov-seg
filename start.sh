#!/bin/bash

conda activate ovseg
# sudo $(which python3) demo.py
sudo nohup $(which python3) demo.py &
tail -f /home/ubuntu/nohup.out