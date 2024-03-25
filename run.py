import torch
import random
from torch import nn, optim
from model import *
from datasets import *
from traintest import *

EPOCH_COUNT = 100
BATCH_SIZE = 1024
LR = 0.001
SHOW_PER_EPOCH = 10
SHOW_IMG_COUNT = 5
PATCH_SIZE = 2
MASK_RATE = 0.75
# define some parameters

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data loader
    train_loader, test_loader = getdataset(BATCH_SIZE)
    _, height, width = train_loader.dataset[0][0].shape
    num_patches = height // PATCH_SIZE * width // PATCH_SIZE
    num_masked_patches = int(num_patches * MASK_RATE)
    mask_params = {
        'height': height,
        'width': width,
        'num_patches': num_patches,
        'mask_id': None,
    }
    # Generate mask parameters

    autoencoder = AutoEncoder()
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    comparison = []
    train_loss_list = []
    test_loss_list = []
    # Initialize the model, optimizer, loss function, and comparison list

    for epoch in range(EPOCH_COUNT):
        mask_params['mask_id'] = random.sample(range(mask_params['num_patches']), num_masked_patches)

        train_loader_masked, train_loader_origin = mask_image(train_loader, mask_params, PATCH_SIZE)
        train_loss = train(
            model, 
            optimizer, 
            criterion, 
            train_loader_masked, 
            train_loader_origin, 
            device
        )
        train_loss_list.append(train_loss)

        test_loader_masked, test_loder_origin = mask_image(test_loader, mask_params, PATCH_SIZE)
        compare, test_loss = test(
            epoch, 
            model, 
            criterion, 
            test_loader_masked, 
            test_loder_origin, 
            device, 
            SHOW_PER_EPOCH, 
            SHOW_IMG_COUNT, 
            BATCH_SIZE
        )
        test_loss_list.append(test_loss)
        comparison.extend(compare)

        print(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')

    visualize(
        comparison, 
        train_loss_list, 
        test_loss_list, 
        SHOW_PER_EPOCH, 
        SHOW_IMG_COUNT, 
        EPOCH_COUNT
    )
