# RL_LDCT: Reinforcement Learning-Based Operator Selection For Low-Dose CT Denoising
Official PyTorch implementation of the paper **RL-LDCT: Reinforcement Learning-Based Operator Selection for Low-Dose CT Denoising**
## Overview
This repository contains the code to reproduce the deep learning model framework for LDCT denoising. By integrating Reinforcement learning, the agent discovered multiple optimal sequences of heterogenous operators (Res-Net, Transformers, State-Space/Mamba, GNN, ConvLSTM).

## Files
- 'blocks_2.py' & model.py : Defines our search space operators and develop the suggested model architecture for the reinforcement learning model.
- train.py, trainer.py & utils.py : Defines files used to train the reinforcement learning agent, generate the subsequent reward and evaluation metrics (PSNR & SSIM) using patch training.
- dataset.py : Loads our dataset for Dicom image slices (.dcm) or min-max normalized Hu saved npy array files.
- dl_model_design.py : Trains our selected model variation independently from the RL model using no crop validation and testing results.
