# Root Segmentation using U-Net

This project implements a deep learning model for segmenting plant root structures from soil in RGB images, using a modified U-Net.
The project was built around the data from: 
  > _Nair, R., Strube, M., Hertel, M., Kolle, O., Rolo, V., & Mirco Migliavacca. (2022). 
  > Automated Minirhizotron Validation Data (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7233828_

The described model expects 512x512 images.

## Architectural Modifications
This implementation extends the classic U-Net with the following enhancements:

#### Group Normalization
Replaces Batch Normalization to support smaller batch sizes.
 - Uses 8 groups by default, following [Wu & He, 2018](https://arxiv.org/abs/1803.08494), which indicates that 8 to 16 channels per group give the best performance.
 - In this model, the smallest layer has 64 channels. 8 Groups = 8 channels / group.

#### Residual Blocks
All convolutional blocks are residual to improve gradient flow and convergence stability.

#### Squeeze-And-Excitation (SE) Blocks
Added to encoder and decoder blocks to enable dynamic channel-wise attention, allowing the network to emphasize informative features.
 - The implemented SE Blocks are as described by [J. Hu et al., 2019](https://arxiv.org/abs/1709.01507)

#### ASPP Bottleneck
The bottleneck is replaced by an ASPP block to enhance multi-scale feature extraction.
 - Especially useful for capturing both fine root hairs and larger root structures.

#### Padding
All convolutions use `padding=1` to both preserve spatial resolution throughout the network and ensure clean skip connections without the need for cropping.

## How to Use

### Training
Training can easily be launched using, the following command :
  `python train.py --data_dir data/1_mesocosm --epochs 100`

Multiple arguments are callable : 
  - \-\-data\_dir to set the data's source directory (default="data/1\_mesocosm")
  - \-\-ckpt\_dir sets the save directory of the checkpoints (default="logs/model\_checkpoints")
  - \-\-batch\_size allows batch size control. (default=4)
  - \-\-epochs sets the number of training epochs. (default=100)
  - \-\-lr controls the learning rate. (default=1e-4)
  - \-\-num\_workers sets the number of data loading workers. (default=2)
  - \-\-save_freq sets the frequency at which checkpoints are saved. (default=33)
    
### Validation
Validation can be used with the following command : 
  `python validate.py --ckpt logs/model_checkpoints/best_model.pt`
