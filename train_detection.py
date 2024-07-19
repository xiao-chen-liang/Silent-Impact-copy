import os
import random
import numpy as np
import tqdm
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arm', default='p')
parser.add_argument('--name', default='test')
parser.add_argument('--split', default='1')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy import signal as sig

from PIL import Image

palette = [
    255, 255, 255,
    255, 153, 153,
    153, 255, 153,
    153, 153, 255,
    255, 255, 153,
    255, 153, 255,
    153, 255, 255,
    204, 153, 153,
    153, 204, 153,
    153, 153, 204,
    204, 204, 153
]

seed = 123456789
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False

from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.arm == 'p':
    data_path = "data/Detection_Passive.pkl"
elif args.arm == 'd':
    data_path = "data/Detection_Dominant.pkl"
else:
    print("Invalid arm")
    exit(1)

log_dir = f"logs/detection/{args.split}"
log_path = f"{log_dir}/{args.name}.txt"
checkpoint_dir = f"checkpoints/detection/{args.split}/{args.name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


num_labels = 2
batch_size = 1
num_epochs = 500
learning_rate = 1e-3


def main():
    data_x, data_y, data_subj = load_pickle(data_path)
    train_subjects, val_subjects, test_subjects = load_split(args.split)

    train_dataset = DetectionDataset(data_x, data_y, data_subj, train_subjects)
    val_dataset = DetectionDataset(data_x, data_y, data_subj, val_subjects)
    test_dataset = DetectionDataset(data_x, data_y, data_subj, test_subjects)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    with open(log_path, 'a') as f:
        f.write(f"\n\nTRAINING\n")
    best_epoch = train(train_dataloader, val_dataloader)

    with open(log_path, 'a') as f:
        f.write(f"\n\nTESTING\n")
    test_acc, test_f1 = test(test_dataloader, best_epoch)
    with open(log_path, 'a') as f:
        f.write(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}\n")


def load_pickle(path):
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pkl.load(f)

    num_items, max_seq_len, _ = data.shape
    max_seq_len -= 1

    # Normalize data
    acc_flat = data[:,1:,:3].reshape(num_items * max_seq_len * 3, 1)
    acc_scaler = MinMaxScaler()
    acc_flat = acc_scaler.fit_transform(acc_flat)
    data[:,1:,:3] = acc_flat.reshape(num_items, max_seq_len, 3)

    gyr_flat = data[:,1:,3:6].reshape(num_items * max_seq_len * 3, 1)
    gyr_scaler = MinMaxScaler()
    gyr_flat = gyr_scaler.fit_transform(gyr_flat)
    data[:,1:,3:6] = gyr_flat.reshape(num_items, max_seq_len, 3)

    # Put data into arrays
    x_array = []
    y_array = []
    subj_array = []
    for item in data:
        item_len = int(item[0, 0])
        item_data = item[1:item_len+1]
        
        item_x = item_data[:, :6].astype(np.float32)
        item_y = item_data[:, 6].astype(np.int64)
        item_y[item_y > 1] = 1

        x_array.append(item_x)
        y_array.append(item_y)
        subj_array.append(int(item[0, 1]))

    x = np.array(x_array, dtype='object')
    y = np.array(y_array, dtype='object')
    subj = np.array(subj_array)

    return x, y, subj


def load_split(split):
    split = int(split)
    subject_order = [17, 12, 13, 20, 8, 19, 15, 9, 7, 11]
    val_subjects = subject_order[(split-1)*2:split*2]
    test_subjects = subject_order[split*2:(split*2)+2] if split < 5 else subject_order[:2]
    train_subjects = [x for x in subject_order if x not in val_subjects and x not in test_subjects]

    return train_subjects, val_subjects, test_subjects


class DetectionDataset(Dataset):
    def __init__(self, data_x, data_y, data_subj, subjects):
        indexes = np.isin(data_subj, subjects)
        self.x = [data_x[i] for i in range(len(data_x)) if indexes[i]]
        self.y = [data_y[i] for i in range(len(data_y)) if indexes[i]]
        
        self.data_len = len(self.x)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(train_dataloader, val_dataloader):
    model = DetectionModel(3, 10, 64, 6, num_labels)
    model = model.to(device)

    class_weights = torch.tensor([1, 5], dtype=torch.float32).to(device)
    CE = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_acc_epoch = 0
    best_f1 = 0
    best_f1_epoch = 0

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = 0
        train_f1 = 0
        correct = 0
        num_train_frames = 0

        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            pred = torch.argmax(out[-1], dim=2, keepdim=False).float()

            loss = 0
            for o in out:
                loss += CE(o.contiguous().view(-1, num_labels), y.long().view(-1))
            train_loss += loss.item()
            
            correct += (pred == y).sum().item()
            train_f1 += f1_score(y.cpu().detach().numpy()[0], pred.cpu().detach().numpy()[0], average='binary', zero_division=1) * x.shape[1]
            num_train_frames += x.shape[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= num_train_frames
        train_acc = correct / num_train_frames
        train_f1 /= num_train_frames

        correct = 0
        f1 = 0
        num_val_frames = 0

        model.eval()
        for i, (x, y) in enumerate(val_dataloader):
            x = x.cuda()
            y = y.cuda()

            out = model(x)[-1]
            pred = torch.argmax(out, dim=2, keepdim=False).float()

            correct += (pred == y).sum().item()
            f1 += f1_score(y.cpu().detach().numpy()[0], pred.cpu().detach().numpy()[0], average='binary', zero_division=1) * x.shape[1]
            num_val_frames += x.shape[1]

        acc = correct / num_val_frames
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch + 1
        f1 /= num_val_frames
        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch + 1
            torch.save(model.state_dict(), f"{checkpoint_dir}/{best_f1_epoch}.ckpt")

        with open(log_path, 'a') as f:
            f.write('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}, Best Acc: {:.4f}, Best Acc Epoch: {}, Val F1: {:.4f}, Best F1: {:.4f}, Best F1 Epoch: {}\n'
                  .format(epoch+1, num_epochs, train_loss, train_acc, acc, best_acc, best_acc_epoch, f1, best_f1, best_f1_epoch))
            
    return best_f1_epoch


def test(test_dataloader, best_epoch):
    model = DetectionModel(3, 10, 64, 6, num_labels)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/{best_epoch}.ckpt"))
    model = model.to(device)

    correct = 0
    f1 = 0
    num_test_frames = 0
    test_np = np.array([], dtype=np.float32).reshape(0, 3)

    model.eval()
    for i, (x, y) in enumerate(test_dataloader):
        x = x.cuda()
        y = y.cuda()
            
        out = model(x)[-1]

        x = x.cpu().detach()
        y = y.cpu().detach()
        out = out.cpu().detach()

        pred = torch.argmax(out, dim=2, keepdim=False).float()
        xyp = torch.stack((x[0,:,0], y[0], pred[0]), dim=1).numpy()
        test_np = np.concatenate((test_np, xyp), 0)
        fig_path = f"{log_dir}/{i}.png"
        save_figure_tiles(test_np, fig_path)

        correct += (pred == y).sum().item()
        f1 += f1_score(y.cpu().detach().numpy()[0], pred.cpu().detach().numpy()[0], average='binary', zero_division=1) * x.shape[1]
        num_test_frames += x.shape[1]

    acc = correct / num_test_frames
    f1 /= num_test_frames

    return acc, f1


def save_figure_tiles(xyp, fig_path):
    max_width = 8000
    sequence_length, _ = xyp.shape

    num_rows = sequence_length // max_width
    if sequence_length % max_width != 0:
        num_rows += 1

    image_list = []
    for ridx in range(num_rows):
        s = ridx * max_width
        e = min(s + max_width, sequence_length)

        gt = xyp[s:e, 1]
        gt = np.tile(gt, (100, 1))
        gt_img = Image.fromarray(gt)
        gt_img = gt_img.convert("P")
        gt_img.putpalette(palette)

        pred = xyp[s:e, 2]
        pred = np.tile(pred, (100, 1))
        pred_img = Image.fromarray(pred)
        pred_img = pred_img.convert("P")
        pred_img.putpalette(palette)

        image_list.append(gt_img)
        image_list.append(pred_img)
    
    small_padding = 1
    big_padding = 10
    final_width = image_list[0].width
    final_height = image_list[0].height * len(image_list) + (small_padding + big_padding) * (len(image_list) // 2) - big_padding
    final_img = Image.new('RGB', (final_width, final_height))
    for i, img in enumerate(image_list):
        start_height = i * img.height
        start_height += small_padding * ((i + 1) // 2)
        start_height += big_padding * (i // 2)
        final_img.paste(img, (0, start_height))
    final_img.save(fig_path)


if __name__ == '__main__':
    main()