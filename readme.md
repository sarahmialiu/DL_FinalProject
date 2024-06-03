# README

Install required packaged from `requirements.txt`. This code was developed under `torch == 1.13.1`, but you are free to use latest version.

Model Architectures adapted for MRI input:
Dimension Reduction U-Net: model_dr_unet.py
Dense U-Net: model_dense_unet.py
Vanilla U-Net: model_unet.py

main.ipynb is used to run each model on dataloaders that are created from a custom class SliceDataset.