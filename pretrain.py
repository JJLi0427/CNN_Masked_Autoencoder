import torch
import numpy as np
import torch
import random
import argparse
from torch import nn, optim
from model import AutoEncoder
from datasets import getdataset, mask_image
from utils import visualize_pretrain, loss_figure, set_logger, set_seed


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--show_per_epoch', type=int, default=10, help='Show per epoch')
    parser.add_argument('--show_img_count', type=int, default=2, help='Show image count')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--mask_rate', type=float, default=0.75, help='Mask rate')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epoch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args


def train(
    model, 
    optimizer, 
    criterion, 
    train_loader_masked, 
    train_loader_origin, 
    device
):
    model.train()
    training_loss = []
    for i, ((mask_data, _), (train_data, _)) in enumerate(zip(train_loader_masked, train_loader_origin)):
        mask_data = mask_data.to(device)
        train_data = train_data.to(device)
        optimizer.zero_grad()               
        _, decoded = model(mask_data)
        loss = criterion(decoded, train_data)           
        loss.backward()                    
        optimizer.step()                 
        training_loss.append(loss.data.cpu().numpy())
        
    avgloss = np.mean(training_loss)
    return avgloss


def test(
    epoch, 
    model, 
    criterion, 
    test_loader_masked, 
    test_loader_origin, 
    device, 
    show_per_epoch, 
    show_img_count, 
    batch_size
):
    model.eval()
    testing_loss = []
    compare = []
    with torch.no_grad():
        for i, ((mask_data, _), (test_data, _)) in enumerate(zip(test_loader_masked, test_loader_origin)):
            mask_data = mask_data.to(device)
            test_data = test_data.to(device)
            _, decoded = model(mask_data)
            loss = criterion(decoded, test_data)
            testing_loss.append(loss.data.cpu().numpy())
            if i == 0 and (epoch == 0 or (epoch+1) % show_per_epoch == 0):
                for j in range(show_img_count):
                    compare_img = torch.cat(
                        [
                            test_data[j:j+1],
                            mask_data[j:j+1], 
                            decoded.view(batch_size, 1, 28, 28)[j:j+1]
                        ]
                    )
                    compare.append(compare_img)
                    # Save the comparison image
    avgloss = np.mean(testing_loss)
    return compare, avgloss


def get_mask_params(batch_img, args):
    _, height, width = batch_img.shape
    num_patches = height // args.patch_size * width // args.patch_size
    num_masked = int(num_patches * args.mask_rate)
    mask_params = {
        'height': height,
        'width': width,
        'num_patches': num_patches,
        'num_masked': num_masked, 
        'mask_id': None,
    }
    # Generate the mask parameters
    return mask_params


def pretrain():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('pretrain')
    logging.info(f'Start training:')
    logging.info(f'Arguments: {args}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    
    # Data loader
    train_loader, test_loader = getdataset(args.batch_size)
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    mask_params = get_mask_params(train_loader.dataset[0][0], args)
    comparison = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epoch):
        mask_params['mask_id'] = random.sample(
            range(mask_params['num_patches']), 
            mask_params['num_masked']
        )

        train_loader_masked, train_loader_origin = mask_image(train_loader, mask_params, args.patch_size)
        train_loss = train(
            model, 
            optimizer, 
            criterion, 
            train_loader_masked, 
            train_loader_origin, 
            device
        )
        train_loss_list.append(train_loss)

        test_loader_masked, test_loder_origin = mask_image(test_loader, mask_params, args.patch_size)
        compare, test_loss = test(
            epoch, 
            model, 
            criterion, 
            test_loader_masked, 
            test_loder_origin, 
            device, 
            args.show_per_epoch, 
            args.show_img_count, 
            args.batch_size
        )
        test_loss_list.append(test_loss)
        comparison.extend(compare)
        
        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
        if (epoch+1) % args.save_every == 0:
            torch.save(model.state_dict(), f'./ckpt/pretrain/{epoch+1}epoch.pth')

    visualize_pretrain(
        comparison, 
        args.show_per_epoch, 
        args.show_img_count, 
    )
    loss_figure(
        train_loss_list, 
        test_loss_list, 
        args.epoch, 
        'pretrain'
    )
    
if __name__ == "__main__":
    pretrain()