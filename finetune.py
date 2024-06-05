import torch
import argparse
import numpy as np
from torch import nn, optim
from datasets import getdataset
from model import AutoEncoder, Classifier
from utils import loss_figure, set_logger, set_seed

def train(
    model, 
    optimizer, 
    criterion, 
    train_loder,
    device
):
    model.train()
    training_loss = []
    for images, labels in train_loder:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.data.cpu().numpy())
        
    avgloss = np.mean(training_loss)
    return avgloss


def test(
    model, 
    criterion, 
    test_loader,
    device, 
):
    model.eval()
    testing_loss = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            testing_loss.append(loss.data.cpu().numpy())
    avgloss = np.mean(testing_loss)
    return avgloss


def init_model(num_classes, model_path):
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder = autoencoder
    
    # freeze the encoder
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False
    model = Classifier(autoencoder.encoder, num_classes)
    return model


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser(description='AutoEncoder')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--model_path', type=str, default='ckpt/pretrain_100ep.pth', help='Model path')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epoch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args


def finetune():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('finetune')
    logging.info(f'Start finetuning:')
    logging.info(f'Arguments: {args}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    
    train_loader, test_loader = getdataset(args.batch_size)
    model = init_model(args.num_classes, args.model_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(args.epoch):
        train_loss = train(
            model, 
            optimizer, 
            criterion, 
            train_loader, 
            device
        )
        test_loss = test(
            model, 
            criterion, 
            test_loader, 
            device
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
        if (epoch+1) % args.save_every == 0:
            torch.save(model.state_dict(), f'ckpt/finetun_{epoch+1}.pth')
            
    loss_figure(
        train_loss_list, 
        test_loss_list, 
        args.epoch, 
        'finetune'
    )

if __name__ == "__main__":
    finetune()