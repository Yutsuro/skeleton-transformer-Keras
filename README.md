# Skeleton Transformer in Keras

Keras implementation of "skeleton transformer module", which is mentioned in [Skeleton-based Action Recognition with Convolutional Neural Networks](https://arxiv.org/abs/1704.07595).

## Install

You can install this module from [PyPI](https://pypi.org/project/skeraton).

```sh
pip install skeraton
```

## How to use

### Module

#### skeraton.SkeletonTransformer

##### Parameters:

All parameters are required.

**timesteps:** Timesteps of input time-series data (equal to number of frames, mentioned as 'T' in the paper)  
**kpts_dim:** Dimentions of keypoints (usually 2 (x, y) or 3 (x, y, z))
**output_dim:** Dimentions of output (mentioned as 'M' in the paper)

##### Input:

**x:** 3-dimentional tensor of shape (batchsize, timesteps, kpts_dim*N) where N is number of joints

