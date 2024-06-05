import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import logging
import time
import random
import numpy as np


def visualize_pretrain(
    comparison,
    show_per_epoch, 
    show_img_count, 
):
    os.makedirs('./figure', exist_ok=True)
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    all_comparisons = torch.cat(comparison, dim=0)
    name = './figure/show_per' + str(show_per_epoch) + 'epoch_' + str(timestamp) + '.png'
    save_image(all_comparisons.cpu(), name, nrow=show_img_count*3)


def loss_figure(
    train_loss_list, 
    test_loss_list, 
    epoch,
    mode
):
    os.makedirs('./figure', exist_ok=True)
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    if mode == 'pretrain':
        title = 'Image MSE Loss'
        figure_path = f'./figure/pretrain_loss_{timestamp}.png'
    elif mode == 'finetune':
        title = 'Cross Entropy Loss'
        figure_path = f'./figure/finetune_loss_{timestamp}.png'
    else:
        raise ValueError('Invalid mode')
    
    plt.plot(range(epoch), train_loss_list, label='train loss')
    plt.plot(range(epoch), test_loss_list, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(figure_path)
    plt.close()
    # Save the loss curve


def set_logger(mode):
    os.makedirs('./ckpt', exist_ok=True)
    
    if mode == 'pretrain':
        filename = './ckpt/pretrain.log'
    elif mode == 'finetune':
        filename = './ckpt/finetune.log'
    else:
        raise ValueError('Invalid mode')

    logging.basicConfig(
        filename=filename, 
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    return logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
