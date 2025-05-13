<div align="center">

<h1>WaveGuard</h1>

<h1>Robust Deepfake Detection and Source Tracing via Dual-Tree Complex Wavelet and Graph Neural Networks</h1>

</div>  <!-- ✅ 这一行必须加，结束居中区域 -->

## ⭐ Abstract


Deepfake technology has great potential in media and entertainment, but it also poses serious risks, including privacy leakage and identity fraud. To address these threats, proactive watermarking methods have emerged by embedding invisible signals for active protection. However, existing approaches are often vulnerable to watermark destruction under malicious distortions, leading to insufficient robustness. Moreover, strong embedding may degrade image quality, making it difficult to balance robustness and imperceptibility.

To solve these problems, we propose WaveGuard, a proactive watermarking framework that explores frequency-domain embedding and graph-based structural consistency optimization. Watermarks are embedded into high-frequency sub-bands using dual-tree complex wavelet transform (DT-CWT) to enhance robustness against distortions and deepfake forgeries. By leveraging joint sub-band correlations, WaveGuard supports robust extraction for source tracing and semi-robust extraction for deepfake detection. We also employ dense connectivity strategies for feature reuse and propose a Structural Consistency Graph Neural Network (SC-GNN) to reduce perceptual artifacts and improve visual quality. Additionally, a Tanh-based Spatial Embedding Attention Module (TSEAM) refines both global and local features, improving watermark concealment without sacrificing robustness.

## 🚀 Method Overview

<div align="center">
    <img width="400" alt="image" src="Image\name01.png">
</div>

The figure shows the comparison of three watermarking embedding strategies: the first two are based on GAN and VAE to optimize the image quality, while WaveGuard introduces GNN structural consistency constraints to enhance invisibility and robustness.

## 📻 Overview

<div align="center">
    <img width="1000" alt="image" src="Image\network.png">
</div>

<div align="center">
Illustration of the overall architecture.
</div>

## 📆 Release Plan

- [x] Project page released
- [x] Dataset preparation instructions released
- [x] Release of core implementation
- [ ] Release of training and evaluation scripts
- [ ] Pretrained model and demo

## 🖥️ Environment Setup

```python
python -m pip install -r requirements.txt
```

## 📁 Dataset Preparation

WaveGuard was trained and tested in CelebA-HQ. We don't own data sets, they can be downloaded from the official website.

- <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" style="color:#0078d4;">Download CelebA-HQ</a>

This project uses the CelebA-HQ dataset with 128×128 and 256×256 resolutions. Please organize images as follows:

```
CelebA-HQ
├── train
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── val
│   └── ...
└── test
    └── ...
```

Ensure all images are cropped and resized appropriately before training.

## 🔧 Training Command

```python
python train.py --config train.yaml
```

## 🧪 Test Command

```python
python test.py 
```

## 🖼️   Visualization

<div align="center">
    <img width="1000" alt="image" src="Image\collage_output.png">
</div>
<div align="center">
Visualization of key modules in the WaveGuard framework.
</div>

## 📬 Contact

If you have any questions, please contact:

- 📧 107552304059@stu.xju.edu.cn  

  
