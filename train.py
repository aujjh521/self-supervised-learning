
#package import
#general
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#pytorch related
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

#pytorch 讀取image好用的功能
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

#從train.py import 需要用到的model class
from pre_train import  Encoder , SupCon, MetricMonitor

#create logger
from Mylog import getMyLogger
logFileName, logDir = 'Mylog_train.log' , 'log'
logger = getMyLogger(logFileName, logDir, 'INFO')

#定義真實training會用到的model
class LinearClassifier(torch.nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 10),
            )

    def forward(self, x):
        x = self.fc(x)
        probs = torch.nn.functional.softmax(x, dim=0)
        return probs
    
#training的early stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'> early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#計算accuracy
def calculate_accuracy(output, target):
    "Calculates accuracy"
    output = output.data.max(dim=1,keepdim=True)[1]
    output = output == 1.0
    output = torch.flatten(output)
    target = target == 1.0
    target = torch.flatten(target)
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item() 

#每個epoch要做的training事情
def training(epoch, model, classifier, train_loader, optimizer, criterion):
    "Training over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        with torch.no_grad():
            features = model.encoder(data)
        output = classifier(features.float())
        loss = criterion(output, labels) 
        accuracy = calculate_accuracy(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        data.detach()
        labels.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']

#每個epoch要做的validation事情
def validation(epoch, model, classifier, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for batch_idx, (data,labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data,labels = data.cuda(), labels.cuda()
            data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
            features = model.encoder(data)
            output = classifier(features.float())
            loss = criterion(output,labels) 
            accuracy = calculate_accuracy(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            data.detach()
            labels.detach()
    logger.info("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']

def main():
    save_file = os.path.join('./results/', 'model.pth')
    use_scheduler = True
    num_epochs = 50
    use_early_stopping = True

    #定義做data augment的流程
    train_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    valid_transform = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    
    #產生資料集 (fine tune用的train, 以及最後考試用的valid)
    train_MNIST_dataset = torchvision.datasets.MNIST('input/',
                        train = True,
                        transform = train_transform,
                        download = True,)

    valid_MNIST_dataset = torchvision.datasets.MNIST('input/',
                        train = False,
                        transform = valid_transform,
                        download = True,)
    
    train_loader = DataLoader(train_MNIST_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_MNIST_dataset, batch_size=64, shuffle=True)

    # Part 2 (接續前面的pre train)
    model = SupCon(Encoder(), head='mlp', feat_dim=128)
    classifier = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    
    ckpt = torch.load(save_file, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    logger.info(f'load pre trained weight finished')

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    train_losses , train_accuracies = [],[]
    valid_losses , valid_accuracies = [],[]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, verbose=True, delta=1e-4, trace_func=logger.info)

    for epoch in range(1, num_epochs+1):
        logger.info(f'start simCLR training for fine tune, epoch {epoch}')
        train_loss, train_accuracy = training(epoch, model, classifier, train_loader, optimizer, criterion)
        valid_loss, valid_accuracy = validation(epoch, model, classifier, valid_loader, criterion)
        
        if use_scheduler:
            scheduler.step()
            
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        #check 是否要early stop
        if use_early_stopping: 
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                logger.info(f'Early stopping at {epoch}')
                #model.load_state_dict(torch.load('checkpoint.pt'))
                break

if __name__ == '__main__':
    main()