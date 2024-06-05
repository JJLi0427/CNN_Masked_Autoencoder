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
    height = mask_params['height']
    width = mask_params['width']
    mask_id = mask_params['mask_id']
    mask_data_loader = []
    copy_data_loader = []
    # Get the mask positions
    mask_positions = [(i // (height // patch_size), i % (width // patch_size)) for i in mask_id]

    for data, label in data_loader:
        masked_image = data.clone()
        for row, col in mask_positions:
            masked_image[
                :, 
                :, 
                row * patch_size:(row + 1) * patch_size, 
                col * patch_size:(col + 1) * patch_size
            ] = 0
        mask_data_loader.append((masked_image, label))
        copy_data_loader.append((data, label))

    return mask_data_loader, copy_data_loader