## Denoising Diffusion Probabilistic Model (DDPM)

This repository contains an implementation of Denoising Diffusion Probabilistic Model (DDPM) along with training and inference scripts.

### Introduction

The Denoising Diffusion Probabilistic Model (DDPM) is a powerful generative model used for image generation and denoising tasks. It leverages the concept of diffusion processes to model the data distribution. In this repository, we implement DDPM using PyTorch.

### Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- NumPy

You can install the required dependencies using pip:

```bash
pip install torch torchvision matplotlib numpy
```

### Forward Diffusion

The forward diffusion process is implemented to add noise iteratively to input images. This is an essential step in training DDPM.

### U-Net Architecture

We utilize a U-Net architecture for denoising images and learning representations. This architecture helps in effectively capturing and utilizing spatial information during the denoising process.

### Loss Function

We define a loss function for training the model, which is derived from mathematical processes involved in diffusion models.

### Dataset

We use the FashionMNIST dataset for training and testing the model. The dataset is preprocessed and transformed to be compatible with the DDPM.

### Training

The model is trained using the Adam optimizer. We iterate over the dataset for multiple epochs, optimizing the model parameters to minimize the defined loss function.

### Sample Forward Diffusion

We provide a visualization of the forward diffusion process, showcasing how noise is progressively added to input images at different timesteps.

### Saving Checkpoints

Checkpoints of the trained model are saved periodically during training, allowing for easy resumption or transfer of training.

### References

- [Diffusion Models Explained](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- Original Paper: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
