# CNN_Autoencoder
This repository contains a PyTorch implementation of an autoencoder for image reconstruction using the MNIST dataset. The autoencoder model is designed to mask images with random patches and then reconstruct the original images from the masked inputs.

## Requirements
Python 3.x
PyTorch
torchvision
numpy
matplotlib

## Usage
1. Clone the repository
2. cd CNN_Autoencoder
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the main script: `python came.py`  
5. Monitor the training process:
* The script will train the autoencoder model over multiple epochs, displaying the training and testing loss for each epoch.  
* Additionally, it will save comparison images showing the original, masked, and reconstructed images.
6. Check the output:
* The comparison images for selected epochs will be saved as show_per_<SHOW_PER_EPOCH>epoch.png.  
* The loss curve plot will be saved as image_loss.png.

## Parameters
* `EPOCH_COUNT`: Number of epochs for training
* `BATCH_SIZE`: Batch size for training
* `LR`: Learning rate for the optimizer
* `SHOW_PER_EPOCH`: Frequency of displaying results per epoch
* `SHOW_IMG_COUNT`: Number of images to display in the comparison
* `PATCH_SIZE`: Size of the image patches for masking
* `MASK_RATE`: Percentage of image patches to mask

## Related Blogs
<https://blog.csdn.net/lijj0304/article/details/136597791>
