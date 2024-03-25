import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

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

def visualize(
    comparison, 
    train_loss_list, 
    test_loss_list, 
    show_per_epoch, 
    show_img_count, 
    epoch_count
):
    all_comparisons = torch.cat(comparison, dim=0)
    name = 'show_per_' + str(show_per_epoch) + 'epoch.png'
    save_image(all_comparisons.cpu(), name, nrow=show_img_count*3)
    plt.plot(range(epoch_count), train_loss_list, label='train loss')
    plt.plot(range(epoch_count), test_loss_list, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Image MSE Loss')
    plt.legend()
    plt.savefig('image_loss.png')
    plt.close()
    # Save the loss curve
