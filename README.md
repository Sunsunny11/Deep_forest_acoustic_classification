# Deep_forest_acoustic_classification
Pytorch: Deep nerual decision forest for acoustic scence classification
# Set up environment
+ Create conda environment with dependencies: **conda env ceate -f requirement.yml**
# Download dataset
+ DCASE2019 development ASC subtask A dataset
+ ESC-50 environmental sound classification dataset
# Run experiments
+ Run: **python train_dNDF.py**
+ Setting paramters: **tree_depth (the depth od tree) and n_tree (the number of tree)**
# Citation
@INPROCEEDINGS{9909575,  <br>
author={Sun, Jianyuan and Liu, Xubo and Mei, Xinhao and Zhao, Jinzheng and Plumbley, Mark D. and Kılıç, Volkan and Wang, Wenwu},  <br>
booktitle={2022 30th European Signal Processing Conference (EUSIPCO)},  <br>
title={Deep Neural Decision Forest for Acoustic Scene Classification},  <br>
year={2022},<br>
pages={772-776}}
