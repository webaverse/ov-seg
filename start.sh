#!/bin/bash

conda activate ov-seg
sudo $(which python3) demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'person' 'water' 'flower' 'mat' 'fog' 'land' 'grass' 'field' 'dirt' 'metal' 'light' 'book' 'leaves' 'mountain' 'tree' 'gravel' 'wood' 'bush' 'bag' 'food' 'path' 'stairs' 'rock' 'house' 'clothes' 'animal' --input ./dalle5.png --output ./pred5 --opts MODEL.WEIGHTS ./ovseg_swinbase_vitL14_ft_mpt.pth