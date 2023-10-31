#!/bin/bash
python run.py \
    --ckpt /home/tony/deep_learning_dev_w_pytorch/module7/models/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 50 \
    --batch_size 12 \
    --cnn_backbone resnet34 \
    --rnn_hidden_size 128 \
    --rnn_n_layers 2 \
    --mode eval
