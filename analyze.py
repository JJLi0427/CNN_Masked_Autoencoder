import torch
import os
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
    parser.add_argument('--model_path', type=str, default='./ckpt/finetune_100ep.pth', help='Model path')
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
    model, device = load_model(args.model_path, args.num_classes)
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
    print(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')



if __name__ == '__main__':
    args = get_args()
    _, test_loader = getdataset(args.batch_size)
    sample(test_loader, args) # Visualize the samples
    score_analyze(test_loader, args) # Analyze the scores