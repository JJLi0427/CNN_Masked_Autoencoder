import torch
import os
from torchvision import datasets, transforms

def getdataset(batch_size):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        ), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        ), 
        batch_size=batch_size, 
        shuffle=True
    )
    return train_loader, test_loader

def mask_image(data_loader, mask_params, patch_size):
    """
    Mask image with random patches
    :param data_loader: data loader
    :param mask_params: dictionary of parameters
    """
    height = mask_params['height']
    width = mask_params['width']
    num_patches = mask_params['num_patches']
    mask_id = mask_params['mask_id']
    mask_data_loader = []
    copy_data_loader = []
    for data, label in data_loader:
        masked_image = torch.zeros_like(data)
        for i in range(num_patches):
            row = i // (height // patch_size)
            col = i % (width // patch_size)
            if i in mask_id:
                masked_image[:, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = 0
            else:
                masked_image[:, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = data[:, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
        mask_data_loader.append((masked_image, label))
        copy_data_loader.append((data, label))
    return mask_data_loader, copy_data_loader