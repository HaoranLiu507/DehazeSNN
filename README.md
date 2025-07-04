# DehazeSNN

> **Abstract:** 
Image dehazing is a critical challenge in computer vision, essential for enhancing image clarity in hazy conditions. 
Traditional methods often rely on atmospheric scattering models, while recent deep learning techniques, specifically 
Convolutional Neural Networks (CNNs) and Transformers, have improved performance by effectively analyzing image 
features. However, CNNs struggle with long-range dependencies, and Transformers demand significant computational 
resources. To address these limitations, we propose DehazeSNN, an innovative architecture that integrates a U-Net-like 
design with Spiking Neural Networks (SNNs). DehazeSNN captures multi-scale image features while efficiently managing
local and long-range dependencies. The introduction of the Orthogonal Leaky-Integrate-and-Fire Block (OLIFBlock) 
enhances cross-channel communication, resulting in superior dehazing performance with reduced computational burden. 
Our extensive experiments show that DehazeSNN is highly competitive to state-of-the-art methods on benchmark datasets,
delivering high-quality haze-free images with a smaller model size and less multiply-accumulate operations.

## Overview

This repository presents **DehazeSNN**, a novel image dehazing architecture that combines U-Net-like design with Spiking Neural Networks (SNNs) to achieve superior dehazing performance while maintaining computational efficiency. Our approach addresses the limitations of traditional CNNs and Transformers by introducing the Orthogonal Leaky-Integrate-and-Fire Block (OLIFBlock) for enhanced cross-channel communication.

### Key Features
- **Efficient Architecture**: U-Net-like design with SNN integration for optimal feature extraction
- **OLIFBlock**: Novel orthogonal mechanism for improved cross-channel communication
- **Multi-scale Processing**: Effective handling of both local and long-range dependencies
- **Computational Efficiency**: Reduced model size and multiply-accumulate operations
- **State-of-the-art Performance**: Competitive results on benchmark datasets

## Demo Results

![](demo1.jpg)
![](demo2.jpg)





## Preparation

### Install

We test the code on PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8.9.0.2

1. Create a new conda environment
```
conda create -n DehazeSNN python=3.11.7
conda activate DehazeSNN
```

2. Install dependencies
```
conda install pytorch=2.1.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Download

You can download the pretrained models and results on  [Zenodo](https://doi.org/10.5281/zenodo.15486831).

The datasets are available for download at [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=3D0) and [RE-HAZE](https://github.com/IDKiro/DehazeFormer).

The final file path should be the same as the following:

```
┬─ saved_models
│   ├─ indoor
│   │   ├─ DehazeSNN-M_best.pth
│   │   └─ ... (model name)
│   └─ ... (dataset name)
└─ datasets
    ├─ indoor
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       └─ ...
    └─ ... (dataset name)
```

## Training and Evaluation

### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python train.py --model (model name) --exp (exp name) --gpu (gpu id) --output (output path) --data_dir (data directory) --dataset (dataset name) --save_dir (save directory) --resume (true/false)
```

For example, we train the DehazeSNN-M on the SOTS indoor set:

```sh
python train.py --model DehazeSNN-M --exp indoor --gpu 2 --output ./output/ --data_dir ./datasets/ --dataset indoor --save_dir ./saved_models/ --resume false
```

### Test

Run the following script to test the trained model:

```sh
python test.py --model (model name) --exp (exp name) --gpu (gpu id) --output (output path) --data_dir (data directory) --dataset (dataset name) --save_dir (save directory)
```

For example, we test the DehazeSNN-M on the RS-HAZE set:

```sh
python test.py --model DehazeSNN-M --exp rshaze --gpu 2 --output ./output/ --data_dir ./datasets/ --dataset rshaze --save_dir ./saved_models/ 
```

## Acknowledgement

This repository is built upon the CUDA C++ kernel implementation from the following paper:
```bibtex
@inproceedings{li2022brain,
  title={Brain-inspired multilayer perceptron with spiking neurons},
  author={Li, Wenshuo and Chen, Hanting and Guo, Jianyuan and Zhang, Ziyang and Wang, Yunhe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={783--793},
  year={2022}
}
```


## Notes

Send email to liuhaoran@cdut.edu.cn if you have critical issues to be addressed.




## Citation

If you find this work useful for your research, please cite our paper:

```bibtex

```
