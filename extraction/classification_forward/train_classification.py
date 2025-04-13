import os
import random
import json
import numpy as np
import tqdm
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arm', default='p')
parser.add_argument('--name', default='test')
parser.add_argument('--split', default='1')
parser.add_argument('--gpu', default='2')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy import signal as sig

seed = 123456789
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False

from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.arm == 'p':
    data_path = "data/combined_numpy_1_2_merge.pkl"
elif args.arm == 'd':
    data_path = "data/Classification_Dominant.pkl"
else:
    print("Invalid arm")
    exit(1)

# get current time to distinguish between logs
import datetime
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
print(f'now: {now}')
log_dir = f"test/{now}/logs/classification/{args.split}"
log_path = f"{log_dir}/{args.name}.txt"
checkpoint_dir = f"test/{now}/checkpoints/classification/{args.split}/{args.name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


num_labels = 4
num_channels = 12
seq_len = 300
batch_size = 16
num_epochs = 100
learning_rate = 1e-4


def main():
    data_x, data_x_fft, data_y, data_subj = load_pickle()
    train_subjects, val_subjects, test_subjects = load_split(args.split)

    train_dataset = PeakDataset(data_x, data_x_fft, data_y, data_subj, train_subjects)
    val_dataset = PeakDataset(data_x, data_x_fft, data_y, data_subj, val_subjects)
    test_dataset = PeakDataset(data_x, data_x_fft, data_y, data_subj, test_subjects)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    with open(log_path, 'w') as f:
        f.write(f"TRAINING\n")
    best_epoch = train(train_dataloader, val_dataloader)

    with open(log_path, 'a') as f:
        f.write(f"\n\nTESTING\n")
    test_acc, test_conf = test(test_dataloader, best_epoch)
    with open(log_path, 'a') as f:
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"{test_conf}\n")


def load_pickle():
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pkl.load(f)

    num_items, window_size, num_channels = data.shape
    window_size -= 1

    # Normalize data
    acc_flat = data[:,1:,:3].reshape(num_items * window_size * 3, 1)
    acc_scaler = MinMaxScaler()
    acc_flat = acc_scaler.fit_transform(acc_flat)
    data[:,1:,:3] = acc_flat.reshape(num_items, window_size, 3)

    gyr_flat = data[:,1:,3:].reshape(num_items * window_size * 3, 1)
    gyr_scaler = MinMaxScaler()
    gyr_flat = gyr_scaler.fit_transform(gyr_flat)
    data[:,1:,3:] = gyr_flat.reshape(num_items, window_size, 3)

    x = data[:,1:].astype(np.float32)
    y = data[:,0,0].astype(np.float32)
    subj = data[:,0,1].astype(np.float32)

    # Fourier transform
    Fs = 120
    low_f = 4
    high_f = 20
    low_c = low_f / (Fs / 2)
    high_c = high_f / (Fs / 2)

    low_b, low_a = sig.butter(6, low_c, 'low')
    med_b, med_a = sig.butter(6, [low_c, high_c], 'bandpass')
    high_b, high_a = sig.butter(6, high_c, 'high')

    x_fft = np.zeros((num_items, 3, window_size, num_channels), dtype=np.float32)
    for i in range(num_items):
        for j in range(num_channels):
            x_fft[i,0,:,j] = sig.filtfilt(low_b, low_a, x[i,:,j]).astype(np.float32)
            x_fft[i,1,:,j] = sig.filtfilt(med_b, med_a, x[i,:,j]).astype(np.float32)
            x_fft[i,2,:,j] = sig.filtfilt(high_b, high_a, x[i,:,j]).astype(np.float32)

    # Convert to torch tensors
    x = torch.Tensor(x)
    x_fft = torch.Tensor(x_fft)
    y = torch.Tensor(y)
    subj = torch.Tensor(subj)

    return x, x_fft, y, subj


def load_split(split):
    with open('splits.json', 'rb') as f:
        dic = json.load(f)

    groups = dic["groups"]
    splits = dic["splits"]

    val_group = splits[split][0]
    test_group = splits[split][1]
    train_group = splits[split][2]

    val_subjects = groups[str(val_group)]
    test_subjects = groups[str(test_group)]
    train_subjects = groups[str(train_group)]
    # train_subjects = [x for x in range(1, 4) if x not in val_subjects and x not in test_subjects]

    return train_subjects, val_subjects, test_subjects


class PeakDataset(Dataset):
    def __init__(self, data_x, data_x_fft, data_y, data_subj, subjects):
        indexes = np.isin(data_subj, subjects)
        self.x = data_x[indexes]
        self.x_fft = data_x_fft[indexes]
        self.y = data_y[indexes]
        
        self.data_len = len(self.x)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx], self.x_fft[idx], self.y[idx]


def train(train_dataloader, val_dataloader):
    model = ClassificationModel(num_channels, num_labels, seq_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1
    best_val_conf = None
    best_epoch = -1

    train_acc_list = []
    val_acc_list = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = 0
        correct = 0
        total = 0
        train_pred = []
        train_true = []

        model.train()
        for i, (x, x_fft, y) in enumerate(train_dataloader):
            x = x.cuda()
            x_fft = x_fft.cuda()
            y = y.cuda()

            out = model(x, x_fft)

            loss = 0
            for i in range(out.shape[0]-1):
                loss += 0.75 * criterion(out[i], y.long().view(-1))
            loss += criterion(out[-1], y.long().view(-1))
            pred = torch.argmax(out[-1], dim=1).float()
    
            correct += (pred == y).sum().item()
            total += y.size(0)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.append(pred.cpu().numpy())
            train_true.append(y.cpu().numpy())

        train_loss /= len(train_dataloader)
        train_acc = correct / total
        train_acc_list.append(train_acc)

        correct = 0
        total = 0
        val_pred = [np.arange(num_labels)]
        val_true = [np.arange(num_labels)]

        model.eval()
        for i, (x, x_fft, y) in enumerate(val_dataloader):
            x = x.cuda()
            x_fft = x_fft.cuda()
            y = y.cuda()

            out = model(x, x_fft)

            if len(out.shape) == 3:
                pred = torch.argmax(out[-1], dim=1).float()
            else:
                pred = torch.argmax(out, dim=1).float()

            correct += (pred == y).sum().item()
            total += y.size(0)

            val_pred.append(pred.cpu().numpy())
            val_true.append(y.cpu().numpy())

        val_acc = correct / total
        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_conf = confusion_matrix(np.concatenate(val_true), np.concatenate(val_pred)) - np.eye(num_labels)
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{checkpoint_dir}/{best_epoch}.ckpt")

        with open(log_path, 'a') as f:
            f.write('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}, Best Val Acc: {:.4f}, Best Epoch: {}\n'
                  .format(epoch+1, num_epochs, train_loss, train_acc, val_acc, best_val_acc, best_epoch))
        
        if epoch == num_epochs - 1:
            train_conf = confusion_matrix(np.concatenate(train_true), np.concatenate(train_pred))
            combined_conf = np.concatenate((
                train_conf, np.ones((train_conf.shape[0], 1)) * 111, 
                best_val_conf), axis=1)
            with open(log_path, 'a') as f:
                f.write(f"{combined_conf}\n")

    return best_epoch

    
def test(test_dataloader, best_epoch):
    model = ClassificationModel(num_channels, num_labels, seq_len)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/{best_epoch}.ckpt"))
    model = model.to(device)

    correct = 0
    total = 0
    test_pred = [np.arange(num_labels)]
    test_true = [np.arange(num_labels)]

    model.eval()
    for i, (x, x_fft, y) in enumerate(test_dataloader):
        x = x.cuda()
        x_fft = x_fft.cuda()
        y = y.cuda()

        out = model(x, x_fft)

        if len(out.shape) == 3:
            pred = torch.argmax(out[-1], dim=1).float()
        else:
            pred = torch.argmax(out, dim=1).float()

        correct += (pred == y).sum().item()
        total += y.size(0)

        test_pred.append(pred.cpu().numpy())
        test_true.append(y.cpu().numpy())

    test_acc = correct / total
    test_conf = confusion_matrix(np.concatenate(test_true), np.concatenate(test_pred)) - np.eye(num_labels)

    return test_acc, test_conf


if __name__ == '__main__':
    main()
