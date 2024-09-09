

import os, random
import csv, torch
import torch.nn as nn
import torch.optim as optim
from utils.get_model_utk import *
from utils.get_data import * 

import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os

from reprogram.adv_reprogram import Adversarial_Reprogramming

def find_min_indices(arr, k):
    if k > len(arr):
        raise ValueError("k cannot be larger than the array size")
    idx = np.argpartition(arr, k)
    sorted_idx = idx[:k].argsort()

    return idx[sorted_idx]

def find_max_indices(arr, k):
    if k > len(arr):
        raise ValueError("k cannot be larger than the array size")

    idx = np.argpartition(arr, -k)[-k:]
    sorted_idx = np.argsort(arr[idx])[::-1]

    return idx[sorted_idx]

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

import argparse

parser = argparse.ArgumentParser(description='Test Adversarial Reprogramming attack in UTKFace scenario')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='simple', type=str, choices=['simple', 'resnet', 'mobilenet', 'transformer'])
parser.add_argument('--original-task', default='age', type=str, help='Specify the original dataset', choices=['age', 'gender', 'race'])
parser.add_argument('--hijack-task', default='race', type=str, help='Specify the hijacking dataset', choices=['age', 'gender', 'race'])
parser.add_argument('--expand', default=1.0, type=float, help='Width expand ratio')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'validate', 'test'])
parser.add_argument('--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
parser.add_argument('--idx', default=0, type=int, help='idx')

run_args = parser.parse_args()

if __name__ == '__main__':

    set_random_seeds(random_seed=int(run_args.seed))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(run_args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = run_args.model # 'resnet' 'transformer' 'mobilenet' 'simple'

    data_dir = './'
    dataset_path = data_dir+'datasets/UTKface.zip'
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    dataset = get_utk_dataset(dataset_path, run_args.original_task, run_args.hijack_task, transform=transform)
    train_loader, val_loader, test_loader, len_train, len_val, len_test = get_dataloader(dataset, batch_size=16)

    classes = {'age': 6, 'gender': 2, 'race': 5}
    
    if model_name == 'simple':
        model = SimpleModel(in_channels=1, num_classes=classes[run_args.original_task], expand=float(run_args.expand))
    elif model_name == 'mobilenet':
        model = MobileNetV2(in_channels=1, num_classes=classes[run_args.original_task], expand=float(run_args.expand))
    elif model_name == 'resnet':
        model = ResNet(in_channels=1, num_classes=classes[run_args.original_task], expand=float(run_args.expand))
    elif model_name == 'transformer':
        model = TransformerModel(in_channels=1, num_classes=classes[run_args.original_task], expand=float(run_args.expand))
    else:
        raise NotImplementedError
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    model, base_acc, base_loss = train_model(model, num_epochs, optimizer, criterion, \
                        train_loader, val_loader, test_loader, len_train, len_val, len_test, device)
    
    print(f"Top-1 Original Task Accuracy: {base_acc:.4f}")

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    dataset = get_utk_dataset(dataset_path, run_args.original_task, run_args.hijack_task, transform=transform)
    train_loader, val_loader, test_loader, len_train, len_val, len_test = get_dataloader(dataset, batch_size=16)


    AR = Adversarial_Reprogramming(args=run_args, net=model, train_loader=train_loader, test_loader=test_loader, label_num=2, task=classes[run_args.hijack_task])

    AR.train()

    # if run_args.mode == 'train':
    #     AR.train()
    # elif run_args.mode == 'validate':
    #     AR.validate()
    # elif run_args.mode == 'test':
    #     AR.test()
    # else:
    #     raise NotImplementedError

    