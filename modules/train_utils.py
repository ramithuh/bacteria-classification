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
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import Specificity

from modules.eval_metrics import *



def train_model(model, data,  criterion, optimizer, scheduler, num_epochs=25, n_classes = 0, device = 'cpu', exp_name = "test", cfg = None):
    dataloaders   = data[0]
    dataset_sizes = data[1]
    class_names   = data[2]
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.mkdir(f'../results/{exp_name}')

    for epoch in range(num_epochs):
        #epoch_wise metrics
        
        train_accuracy = Accuracy(average = None, num_classes = n_classes, compute_on_step=False).to(device)
        val_accuracy   = Accuracy(average = None, num_classes = n_classes, compute_on_step=False).to(device)
        
        if(n_classes == 2): ## Calculate *binary* classification metrics
            train_f1 = F1Score(multiclass=False, compute_on_step=False).to(device)
            val_f1   = F1Score(multiclass=False, compute_on_step=False).to(device)

            train_precision = Precision(multiclass=False, compute_on_step=False).to(device)
            val_precision   = Precision(multiclass=False, compute_on_step=False).to(device)

            train_recall = Recall(multiclass=False, compute_on_step=False).to(device)
            val_recall   = Recall(multiclass=False, compute_on_step=False).to(device)

            train_specificity = Specificity(multiclass=False, compute_on_step=False).to(device)
            val_specificity   = Specificity(multiclass=False, compute_on_step=False).to(device)


        train_preds = torch.empty([0, ])
        train_labels = torch.empty([0, ])

        val_preds = torch.empty([0, ])
        val_labels = torch.empty([0, ])

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
                       
                        if(n_classes == 2): #update binary classification metrics (train)
                            train_f1(preds, labels)
                            train_precision(preds, labels)
                            train_recall(preds, labels)
                            train_specificity(preds, labels)

                        train_preds  = torch.cat((train_preds, preds.cpu()), dim = 0)
                        train_labels = torch.cat((train_labels, labels.cpu()), dim = 0)

                        loss.backward()
                        optimizer.step()
                    else:
                        val_accuracy(preds, labels) #update val accuracy

                        if(n_classes == 2): #update binary classification metrics (val)
                            val_f1(preds, labels)
                            val_precision(preds, labels)
                            val_recall(preds, labels)
                            val_specificity(preds, labels)
                        
                        val_preds  = torch.cat((val_preds, preds.cpu()), dim = 0)
                        val_labels = torch.cat((val_labels, labels.cpu()), dim = 0)

                        

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

        t_f1 = float(train_f1.compute()) if n_classes == 2 else float('nan')
        v_f1 = float(val_f1.compute()) if n_classes == 2 else float('nan')

        t_precision = float(train_precision.compute()) if n_classes == 2 else float('nan')
        v_precision = float(val_precision.compute()) if n_classes == 2 else float('nan')

        t_recall = float(train_recall.compute()) if n_classes == 2 else float('nan')
        v_recall = float(val_recall.compute()) if n_classes == 2 else float('nan')

        t_specificity = float(train_specificity.compute()) if n_classes == 2 else float('nan')
        v_specificity = float(val_specificity.compute()) if n_classes == 2 else float('nan')
        
        print(t_acc)
        print()
        print(v_acc)
                
        train_data = [[name, prec] for (name, prec) in zip(class_names, t_acc)]
        train_table = wandb.Table(data=train_data, columns=["class_name", "accuracy"])      
        
        val_data = [[name, prec] for (name, prec) in zip(class_names, v_acc)]
        val_table = wandb.Table(data=val_data, columns=["class_name", "accuracy"])           
        
        train_confusion_matrix, _  = get_confusion_matrix(train_preds, train_labels, n_classes, class_names)
        val_confusion_matrix, _   = get_confusion_matrix(val_preds, val_labels, n_classes, class_names)

        wandb.log({"train loss" : train_loss, "train accuracy" : train_acc ,
                   "val loss" : val_loss, "val accuracy" : val_acc, 

                   "train f1" : t_f1, "val f1" : v_f1,
                   "train precision" : t_precision, "val precision" : v_precision,
                   "train recall" : t_recall, "val recall" : v_recall,
                   "train specificity" : t_specificity, "val specificity" : v_specificity,
                   
                   "train class accuracies": wandb.plot.bar(train_table, "class_name" , "accuracy", title="Train Per Class Accuracy"),
                   "val class accuracies": wandb.plot.bar(val_table, "class_name" , "accuracy", title="Val Per Class Accuracy"),
                   
                   "train_confusion_matrix" : train_confusion_matrix,
                   "val_confusion_matrix" : val_confusion_matrix})
        print()

        save_model_name =  f"../results/{exp_name}/latest_model_epoch-{epoch}.pth"
        torch.save({
                'state_dict': model.state_dict(),
                'cfg': cfg,
                'epoch': epoch}, save_model_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model