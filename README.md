# Automatic segmentation of NPC GTV from CT images

This repository proivdes source code for automatic segmentation of Gross Target Volume (GTV) of Nasopharynx Cancer (NPC) from CT images according to the following paper:

* [1] Haochen Mei, Wenhui Lei, Ran Gu, Shan Ye, Zhengwentai Sun, Shichuan Zhang and Guotai Wang. "Automatic Segmentation of Gross Target Volume of Nasopharynx Cancer using Ensemble of Multiscale Deep Neural Networks with Spatial Attention." NeuroComputing (under review). 2020.

# Requirement
* Pytorch version >=0.4.1
* TensorboardX to visualize training performance
* Some common python packages such as Numpy, Pandas, SimpleITK

# Usage
Run the following command to install PyMIC:

```bash
pip install PYMIC
```
In this example, we use 2D U-Net to segment the heart from X-Ray images. First we download the images from internet, then edit the configuration file for training and testing. During training, we use tensorboard to observe the performance of the network at different iterations. We then apply the trained model to testing images and obtain quantitative evaluation results.
