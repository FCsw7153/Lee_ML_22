import math
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image

class Dataset(Dataset):
    def __init__(self,x ,y):
        self.data = x
        self.label = y
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
    
class CNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
    
    def forward(self, x):
        x = self.feature(x)

        x = self.classifier(x)

        return x

def semi_train(dataset, model, config ,threshold = 0.65):
    model.eval()

    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle = True, pin_memory = False)

    softmax = nn.Softmax(dim = -1)

    for batch in tqdm(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)

    return dataset

def trainer(train_dataloader, valid_dataloader, unlabeled_set, config, model, device, do_semi):
    criterion = nn.CrossEntropyLoss()

    optimizater = torch.optim.Adam(model.parameters(), lr = config['lr'],  weight_decay=1e-5)

    n_epoch, best_loss = config['n_epoch'], math.inf

    if do_semi:
        unlabeled_set = semi_train(unlabeled_set, model)
        concat_dataset = ConcatDataset([train_set, unlabeled_set])
        train_loader = DataLoader(concat_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    for epoch in range(n_epoch):
        model.train()
        train_loss_record = []
        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizater.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizater.step()
            train_loss_record.append(loss)
        train_loss = sum(train_loss_record) / len(train_loss_record)
        print(f"Epoch: {epoch+1}/{n_epoch}, Train Loss: {train_loss}")

        model.eval()
        valid_loss_record = []
        correct_predictions = 0
        total_samples = 0
        for x, y in tqdm(valid_dataloader, desc = f"Epoch: {epoch+1}/{n_epoch} [Valid]"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                valid_loss_record.append(loss)
                _, predicted_labels = torch.max(pred, 1)
                total_samples += y.size(0)
                correct_predictions += (predicted_labels == y).sum().item()
        valid_loss = sum(valid_loss_record) / len(valid_loss_record)  
        accuracy = 100 * correct_predictions / total_samples  
        print(f"Epoch {epoch+1}/{n_epoch}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # if best_loss > valid_loss:
    #     best_loss = valid_loss
    #     torch.save(model.state_dict(), config['save_dir'])
    #     print('Saving model with loss {:.3f}...'.format(best_loss))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),

        # 随机旋转 (-15 到 15 度)
        transforms.RandomRotation(15),
        
        # 随机小幅平移
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),

        transforms.ToTensor()
    ])

    config = {
        'lr': 0.0001,
        'n_epoch': 80,
        'save_dir': "./models/best_model.pth",
        'batch_size': 128,
    }

    train_set = DatasetFolder("/root/shared-nvme/ml/dataset/hw3/food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=transform)
    valid_set = DatasetFolder("/root/shared-nvme/ml/dataset/hw3/food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=transform)
    unlabeled_set = DatasetFolder("/root/shared-nvme/ml/dataset/hw3/food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=transform)
    test_set = DatasetFolder("/root/shared-nvme/ml/dataset/hw3/food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_transform)

    # Construct data loaders.
    train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    valid_dataloader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
    
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # print(train_data[0])
    # print(train_label[0])

    model = CNN(input_channels = 3).to(device)

    trainer(train_dataloader, valid_dataloader, unlabeled_set, config, model, device, True)

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_dataloader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, labels = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    # Save predictions into the file.
    with open("predict.csv", "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")