#!/bin/bash

frame_lengths=(0.5 False)
model_names=("lightrespidet" "resnet18" "resnet50" "vgg16" "m34_res"  "efficientb0")

for frame_length in "${frame_lengths[@]}"; do
    for model_name in "${model_names[@]}"; do
        echo "Running with frame_length=${frame_length}, model_name=${model_name}"
        python main_final.py '../configs/best_model.ini' --frame_length ${frame_length} --model_name ${model_name}
    done
done