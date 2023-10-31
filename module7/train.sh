#!/bin/bash
python run.py \
    --frame_dir /home/tony/deep_learning_dev_w_pytorch/data/UCF50 \
    --train_size 0.70 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 128 \
    --rnn_n_layers 1 \
    --dropout 0.2 \
    --n_epochs 100 \
    --mode train
