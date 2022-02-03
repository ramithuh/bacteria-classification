from __future__ import print_function, division
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb

# from modules.helpers import *
# from modules.datasets import *
# from modules.dataloaders import *
from torchsummary import summary
from torchmetrics.classification import Accuracy



def train_model(model, data,  criterion, optimizer, scheduler, num_epochs=25, n_classes = 0, device = 'cpu'):
    dataloaders   = data[0]
    dataset_sizes = data[1]
    class_names   = data[2]
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_accuracy = Accuracy(average = None, num_classes = n_classes, compute_on_step=False).to(device)
        val_accuracy = Accuracy(average = None, num_classes = n_classes, compute_on_step=False).to(device)
            
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase        
        train_loss = 0
        train_acc = 0 
    
        val_loss = 0
        val_acc = 0 
        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device,dtype=torch.float)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_accuracy(preds, labels) #update train accuracy
                        loss.backward()
                        optimizer.step()
                    else:
                        val_accuracy(preds, labels) #update val accuracy
                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                        
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                scheduler.step()
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        t_acc = train_accuracy.compute().tolist()
        v_acc = val_accuracy.compute().tolist()
        
        print(t_acc)
        print()
        print(v_acc)
                
        train_data = [[name, prec] for (name, prec) in zip(class_names, t_acc)]
        train_table = wandb. Table (data=train_data, columns=["class_name", "accuracy"])      
        
        val_data = [[name, prec] for (name, prec) in zip(class_names, v_acc)]
        val_table = wandb. Table (data=val_data, columns=["class_name", "accuracy"])           
        
        wandb.log({"train loss" : train_loss, "train accuracy" : train_acc ,
                   "val loss" : val_loss, "val accuracy" : val_acc, 
                   
                   "train class accuracies": wandb.plot.bar(train_table, "class_name" , "accuracy", title="Train Per Class Accuracy"),
                   "val class accuracies": wandb.plot.bar(val_table, "class_name" , "accuracy", title="Val Per Class Accuracy")})
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model