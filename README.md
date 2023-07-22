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

Please cite us:
```
@inproceedings{10.1145/3543507.3583250,
author = {Li, Yingjie and Zhao, Chenye and Caragea, Cornelia},
title = {TTS: A Target-Based Teacher-Student Framework for Zero-Shot Stance Detection},
year = {2023},
isbn = {9781450394161},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3543507.3583250},
doi = {10.1145/3543507.3583250},
abstract = {The goal of zero-shot stance detection (ZSSD) is to identify the stance (in favor of, against, or neutral) of a text towards an unseen target in the inference stage. In this paper, we explore this problem from a novel angle by proposing a Target-based Teacher-Student learning (TTS) framework. Specifically, we first augment the training set by extracting diversified targets that are unseen during training with a keyphrase generation model. Then, we develop a teacher-student framework which effectively utilizes the augmented data. Extensive experiments show that our model significantly outperforms state-of-the-art ZSSD baselines on the available benchmark dataset for this task by 8.9\% in macro-averaged F1. In addition, previous ZSSD requires human-annotated targets and labels during training, which may not be available in real-world applications. Therefore, we go one step further by proposing a more challenging open-world ZSSD task: identifying the stance of a text towards an unseen target without human-annotated targets and stance labels. We show that our TTS can be easily adapted to the new task. Remarkably, TTS without human-annotated targets and stance labels even significantly outperforms previous state-of-the-art ZSSD baselines trained with human-annotated data. We publicly release our code 1 to facilitate future research.},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {1500–1509},
numpages = {10},
keywords = {stance detection, data augmentation, zero-shot learning},
location = {Austin, TX, USA},
series = {WWW '23}
}
```
