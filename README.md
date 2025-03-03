<div align="center">
<h1> RoMERPA-UNet: Robust Medical Image Segmentation with Multi-frequency Edge Refinement and Prompt-guided Attention </h1>
</div>

## 🎈 News

- [2025.2.19] Training and inference code released

## ⭐ Abstract

Medical image segmentation faces numerous challenges, particularly in exploring multi-scale and multi-frequency features for effective edge detection and noise suppression. Plus, enhancing the model's adaptability and generalization ability in diverse pathological images is an urgent issue. 
To address problems, we propose RoMERPA-UNet, which incorporates two key modules: the Edge Detection Block (EDB) and the Hybrid Prompt Block (HPB). 
The EDB enhances feature maps via multi-scale convolutions and separates multi-frequency information. High-frequency components are used to capture boundaries of salient objects, while low-frequency components help suppress noise introduced by non-salient objects. Meanwhile, by combining multi-directional with fine-grained global-local offsets, the model's adaptability to irregular edges is optimized. 
The HPB improves generalization performance by generating high-frequency and low-frequency prompt masks and combining them with prompt-guided cross-attention to extract transferable segmentation features applicable to various medical cases. 
Evaluation results on seven public datasets indicate that RoMERPA-UNet outperforms eleven existing advanced methods in segmentation accuracy. 

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="asserts/challen.png?raw=true">
</div>

The challenges: Medical images of different pathologies exhibit significant differences, with the complex edges and noise interference.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/network.png?raw=true">
</div>

Illustration of the overall architecture of RoMERPA-UNet. (I) EPFM is Edge-Prompt Fusion Module, (II) EDB is Edge Detection Block, (III) HPB is Hybrid Prompt Block, (II.a) MSFA is Multi-scale Feature Aggressiveness, (II.b) MFFE is Multi-frequency Feature Extraction, (II.c) CFR is Contour Feature Refinement, and (III.a) PGC is Prompt-guided Cross-attention.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n RoMERPAUNet python=3.10
conda activate RoMERPAUNet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), and PH2 from this [link](https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), BUSI from this [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), CVC-ClinkDB from this [link](https://www.kaggle.com/datasets/balraj98/cvcclinicdb?resource=download), Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018), and COVID-19 from this [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the RoMERPA-UNet

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the RoMERPA-UNet

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/Visualization.png?raw=true">
</div>

Visualization results of twelve state-of-the-art methods and RoMERPA-UNet for different lesions. The red circles indicate areas of incorrect predictions.

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="asserts/compara.png?raw=true">
</div>

<div align="center">
    We compare our method against twelve state-of-the-art methods, evaluating segmentation performance on the ISIC2018, Kvasir, Monu-Seg, COVID-19, and BUSI datasets, and assessing generalization on the PH2 and CVC-ClinkDB datasets.
</div>

## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveESWA/RoMERUNet/blob/main/LICENSE).

