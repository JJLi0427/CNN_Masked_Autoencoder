import torch
import os
import re
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from datasets import getdataset
from model import AutoEncoder, Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--model_path', type=str, default='./ckpt/finetune/100epoch.pth', help='Model path')
    parser.add_argument('--model_dir', type=str, default='./ckpt/finetune/', help='Model directory')
    args = parser.parse_args()
    return args


def load_model(model_path, num_calsses):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    autoencoder = AutoEncoder()
    model = Classifier(autoencoder.encoder, num_calsses).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device


def sample(test_loader, args):
    os.makedirs('./figure', exist_ok=True)
    model, device = load_model(args.model_path, args.num_classes)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(8):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        ax[i//4, i%4].imshow(img, cmap='gray')
        ax[i//4, i%4].set_title(f'True: {labels[i].item()}, Predicted: {predicted[i].item()}')
        ax[i//4, i%4].axis('off')

    plt.savefig(f'./figure/samples.png')
    plt.close()

def score_analyze(test_loader, args):
    model_list = [file for file in os.listdir(args.model_dir) if file.endswith('.pth')]
    model_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    epoch_list = []  # New list to store the epoch of each model
    
    for model_path in tqdm(model_list):
        model, device = load_model(args.model_dir + model_path, args.num_classes)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        epoch_list.append(int(re.findall(r'\d+', model_path)[0]))  
        # Calculate the epoch of the model and add it to the list
    
    os.makedirs('./figure', exist_ok=True)
    plt.plot(epoch_list, accuracy_list, label='accuracy')
    plt.plot(epoch_list, precision_list, label='precision')
    plt.plot(epoch_list, recall_list, label='recall')
    plt.plot(epoch_list, f1_list, label='f1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score Analysis')
    plt.legend()
    plt.savefig('./figure/score_analysis.png')
    plt.close()


if __name__ == '__main__':
    args = get_args()
    _, test_loader = getdataset(args.batch_size)
    sample(test_loader, args) # Visualize the samples
    score_analyze(test_loader, args) # Analyze the scores