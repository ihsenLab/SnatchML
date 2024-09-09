
import os, random
import csv, torch
import torch.nn as nn
import torch.optim as optim
from utils.get_data import MultiLabelTestDataset
from utils.get_model_pneu import *
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

parser = argparse.ArgumentParser(description='Test Adversarial Reprogramming attack in Pneumonia scenario')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='simple', type=str, choices=['simple', 'resnet', 'mobilenet', 'transformer'])
parser.add_argument('--setting', default='black', type=str, help='Specify the attack setting', choices=['black', 'white'])
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
    
    data_dir = '/home/hbouzidi/hbouzidi/'

    path = data_dir+"datasets/chest_xray/"
    transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    categories = ['train','val','test']
    dset = {x : torchvision.datasets.ImageFolder(path+x, transform=transformers) for x in categories}

    dataset_sizes = {x : len(dset[x]) for x in categories}
    dataloaders =  {x : torch.utils.data.DataLoader(dset[x], batch_size=16, shuffle=True, num_workers=0) for x in categories}

    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    len_train, len_val, len_test = dataset_sizes['train'], dataset_sizes['val'], dataset_sizes['test']

    if model_name == 'simple':
        model = SimpleModel(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'mobilenet':
        model = MobileNetV2(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'resnet':
        model = ResNet(in_channels=3, num_classes=2, expand=float(run_args.expand))
    elif model_name == 'transformer':
        model = TransformerModel(in_channels=3, num_classes=2, expand=float(run_args.expand))
    else:
        raise NotImplementedError
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    model, base_acc, base_loss = train_model(model, num_epochs, optimizer, criterion, \
                        train_loader, val_loader, test_loader, len_train, len_val, len_test, device)
    
    print(f"Top-1 Original Task Accuracy: {base_acc:.4f}")

    transformers = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(64, 64)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_path = os.path.join(path, 'train')
    skip_class = 'NORMAL'
    multi_label_train_dataset = MultiLabelTestDataset(train_path, transform=transformers, skip_class=skip_class)
    multi_label_train_loader = DataLoader(multi_label_train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    test_path = os.path.join(path, 'test')
    skip_class = 'NORMAL'
    multi_label_test_dataset = MultiLabelTestDataset(test_path, transform=transformers, skip_class=skip_class)
    multi_label_test_loader = DataLoader(multi_label_test_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

    AR = Adversarial_Reprogramming(args=run_args, net=model, train_loader=multi_label_train_loader, test_loader=multi_label_test_loader, label_num=2)

    AR.train()

    # if run_args.mode == 'train':
    #     AR.train()
    # elif run_args.mode == 'validate':
    #     AR.validate()
    # elif run_args.mode == 'test':
    #     AR.test()
    # else:
    #     raise NotImplementedError

    