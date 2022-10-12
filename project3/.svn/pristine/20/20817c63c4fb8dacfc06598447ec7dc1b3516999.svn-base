import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
from models import *
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import MultiStepLR
np.set_printoptions(suppress=True)


def set_seed(SEED=0):
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    # ---
    # Your code goes here
    pass
    # ---


def train(model, args, train_loader, test_loader):
    EPOCHS = 80
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    criterion = nn.MSELoss()
    model.train()
    for epoch in tqdm(range(1, EPOCHS+1), total=EPOCHS):
        total_loss = 0
        for i, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            y_pred = model(features)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        total_loss /= len(train_loader)
        test_loss = test(model, test_loader)
        model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'
        print("Train Loss: {:.8f}, Test Loss: {:.8f}".format(total_loss, test_loss))
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
    # ---
    # Your code goes here
    # ---


def test(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    # --
    # Your code goes here
    test_loss = 0
    for i, (features, labels) in enumerate(test_loader):
        with torch.no_grad():
            y_pred = model(features)
        loss = criterion(y_pred, labels)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    set_seed()
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8)

    # ---
    # Your code goes here
    # ---
    model = build_model(args.num_links, 0.01)
    train(model, args, train_loader,test_loader)
    

if __name__ == '__main__':
    main()
