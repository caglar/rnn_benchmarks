#!/bin/bash -e
CUDA_VISIBLE_DEVICES=2 python ptb_word_lm.py --data_path=/home/benchmark/code/rnn_exps/torch/ptb/lstm/data --model small
