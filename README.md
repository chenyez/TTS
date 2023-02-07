# TTS

This is the repository for the paper: ***"TTS: A Target-based Teacher-Student Framework for Zero-Shot
Stance Detection"***

Our code is developed based on python 3.8, PyTorch 1.10.1, CUDA 11.1. Experiments are performed on a single NVIDIA RTX A5000 GPU.

To run TTS for ZSSD task:
```
cd ./TTS_zeroshot/src
```
For 10% training setting:
```
nohup bash ./train_LEDaug_BART_10train_tune_tensorboard_5.sh > train_LEDaug_BART_10train_tune_tensorboard_5_results.log 2>&1 & 
```
For 100% training setting:
```
nohup bash ./train_LEDaug_BART_100train_tune_tensorboard_5.sh > train_LEDaug_BART_100train_tune_tensorboard_5_results.log 2>&1 &
```

To run TTS for open-world ZSSD task:
```
cd ./TTS_openworld/src
```
