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
# from torchmetrics.classification import MulticlassAUROC
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import Specificity
from torchmetrics import AUROC


from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from modules.eval_metrics import *


def test_model(model, data,  criterion, n_classes = 0, device = 'cpu', cfg = None):
    dataloaders   = data[0]
    dataset_sizes = data[1]
    class_names   = data[2]
    
    since = time.time()

    test_accuracy = Accuracy(average = None, num_classes = n_classes, compute_on_step=False).to(device)
    
    if(n_classes == 2): ## Calculate *binary* classification metrics
        test_f1 = F1Score(task="binary", compute_on_step=False).to(device)
    
        test_precision = Precision(task="binary", compute_on_step=False).to(device)
        
        test_recall = Recall(task="binary", compute_on_step=False).to(device)
        
        test_specificity = Specificity(task="binary", compute_on_step=False).to(device)

    test_preds = torch.empty([0, ])
    test_labels = torch.empty([0, ])

    print('starting testing..')
      
    test_loss = 0
    test_acc = 0 

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
            
    for inputs, labels in dataloaders["test"]:
        print(".", end = '')

        inputs = inputs.to(device,dtype=torch.float)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        test_accuracy(preds, labels) #update test accuracy

        if(n_classes == 2): #update binary classification metrics (test)
            test_f1(preds, labels)
            test_precision(preds, labels)
            test_recall(preds, labels)
            test_specificity(preds, labels)
                
        test_preds  = torch.cat((test_preds, preds.cpu()), dim = 0)
        test_labels = torch.cat((test_labels, labels.cpu()), dim = 0)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
                        

    epoch_loss = running_loss / dataset_sizes["test"]
    epoch_acc = running_corrects.double() / dataset_sizes["test"]
                
    test_loss = epoch_loss
    test_acc = epoch_acc

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        "test", epoch_loss, epoch_acc))


    t_acc = test_accuracy.compute().tolist()
    t_f1 = float(test_f1.compute()) if n_classes == 2 else float('nan')
    t_precision = float(test_precision.compute()) if n_classes == 2 else float('nan')
    t_recall = float(test_recall.compute()) if n_classes == 2 else float('nan')
    t_specificity = float(test_specificity.compute()) if n_classes == 2 else float('nan')
    
    print(t_acc)
    print()
                
    test_data = [[name, prec] for (name, prec) in zip(class_names, t_acc)]
    test_table = wandb.Table(data=test_data, columns=["class_name", "accuracy"])      
    
    test_confusion_matrix, saved_confmatrix = get_confusion_matrix(test_preds, test_labels, n_classes, class_names)

    wandb.log({"test loss" : test_loss, "test accuracy" : test_acc ,
                "test f1" : t_f1,
                "test precision" : t_precision,
                "test recall" : t_recall,
                "test specificity" : t_specificity,
        
                "test class accuracies": wandb.plot.bar(test_table, "class_name" , "accuracy", title="Test Per Class Accuracy"),
                "test_confusion_matrix" : test_confusion_matrix
                })
    print()

    time_elapsed = time.time() - since
    print('testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {:4f}'.format(test_acc))

    return saved_confmatrix




def test_model_in_groups(model, data,  criterion, n_classes = 0, device = 'cpu', cfg = None):
    dataloaders   = data[0]
    dataset_sizes = data[1]
    class_names   = data[2]
    N = data[3]
    
    since = time.time()

    
    
    if(n_classes == 2): ## Calculate *binary* classification metrics
        test_accuracy = Accuracy(task="binary", average = None, num_classes = n_classes, compute_on_step=False).to(device)

        test_f1        = F1Score(task="binary", compute_on_step=False).to(device)
        test_precision = Precision(task="binary", compute_on_step=False).to(device)
        test_recall    = Recall(task="binary", compute_on_step=False).to(device)
        test_specificity = Specificity(task="binary", compute_on_step=False).to(device)
        # test_auroc     = AUROC(task="binary").to(device)
    else:
        test_accuracy = Accuracy(task="multiclass", average = None, num_classes = n_classes, compute_on_step=False).to(device)

        test_f1          =      F1Score(task="multiclass", num_classes = n_classes, compute_on_step=False, average = None).to(device)
        test_precision   =    Precision(task="multiclass", num_classes = n_classes, compute_on_step=False, average = None).to(device)
        test_recall      =       Recall(task="multiclass", num_classes = n_classes, compute_on_step=False, average = None).to(device)
        test_specificity =  Specificity(task="multiclass", num_classes = n_classes, compute_on_step=False, average = None).to(device)
        # test_auroc       = MulticlassAUROC(num_classes = n_classes, average="macro").to(device)

    test_preds = torch.empty([0, ])
    test_labels = torch.empty([0, ])
    test_labels_ = torch.empty([0, ])

    print('starting group testing..')
    print(f' dataloader has {len(class_names)} classes to be evaluated')
      
    test_loss = 0
    test_acc = 0 

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    
    outs = {}
    pred_probs = torch.empty([0, ])
    softmax = nn.Softmax(dim=1)

    for i in range(0, len(dataloaders)): #loop through each strain of dataloader
        print(f"New strain batch eval - {i}", end = '\n')

        for inputs, labels in dataloaders[str(i)]: #take a batch of data from each strain

            inputs = inputs.to(device,dtype=torch.float)
            labels_ = labels.to(device) #same label map without getting mode
            labels = labels.to(device)

            outputs = model(inputs)
            

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            preds = torch.mode(preds, 0)[0]
            preds = torch.reshape(preds, (-1,))
            
            labels = torch.mode(labels, 0)[0]
            labels = torch.reshape(labels, (-1,))

            if(labels.cpu().numpy()[0] not in outs):
                outs[labels.cpu().numpy()[0]] = []

            probabilities = softmax(outputs.detach().cpu()).tolist() #convert to probabilities

            outs[labels.cpu().numpy()[0]].append(probabilities) #append probabilities to the dict of probabilities of each class

            test_accuracy(preds, labels) #update test accuracy

            #if(n_classes == 2): #update binary classification metrics (test)
            test_f1(preds, labels)
            test_precision(preds, labels)
            test_recall(preds, labels)
            test_specificity(preds, labels)
                    
            test_preds  = torch.cat((test_preds, preds.cpu()), dim = 0)
            test_labels = torch.cat((test_labels, labels.cpu()), dim = 0)
            test_labels_ = torch.cat((test_labels_, labels_.cpu()), dim = 0)
            pred_probs  = torch.cat((pred_probs, softmax(outputs.detach().cpu())), dim = 0)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
                        

    epoch_loss = running_loss / dataset_sizes["test"]
    epoch_acc = running_corrects.double() / dataset_sizes["test"]
                
    test_loss = epoch_loss
    test_acc = epoch_acc

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        "test", epoch_loss, epoch_acc))


    ##AUROC
    t_auroc = 0 #test_auroc(pred_probs, test_labels_.to(dtype = torch.int32))
    ### AUROC from sklearn

    # print("shape of pred_probs", pred_probs.shape)
    # print("shape of test_labels", test_labels_.shape)


    # print("Sklearn ROC AUC (ovr)", roc_auc_score(test_labels_.to(dtype = torch.int32), pred_probs, multi_class= 'ovr'))
    # print("Sklearn ROC AUC (ovo)", roc_auc_score(test_labels_.to(dtype = torch.int32), pred_probs, multi_class= 'ovo'))

    t_acc = test_accuracy.compute().tolist()
    t_f1 = float(test_f1.compute()) if n_classes == 2 else test_f1.compute().tolist()
    t_precision = float(test_precision.compute()) if n_classes == 2 else test_precision.compute().tolist()
    t_recall = float(test_recall.compute()) if n_classes == 2 else test_recall.compute().tolist()
    t_specificity = float(test_specificity.compute()) if n_classes == 2 else test_specificity.compute().tolist()
    
    print("test accuracy",t_acc)
    print("test f1",t_f1)
    print("test precision",t_precision)
    print("test recall",t_recall)
    print("test specificity",t_specificity)

    print("scikit learn metrics")
    print(classification_report(test_labels, test_preds))
                

    ## Accuracy Table
    test_acc_data  = [[name, prec] for (name, prec) in zip(class_names, t_acc)]
    test_acc_table = wandb.Table(data=test_acc_data, columns=["class_name", "accuracy"])

    ## F1 Table
    if(n_classes != 2):
        test_f1_data  = [[name, prec] for (name, prec) in zip(class_names, t_f1)]
        test_f1_table = wandb.Table(data=test_f1_data, columns=["class_name", "f1"])
    else:
        test_f1_table = wandb.Table()

    ## Precision Table
    if(n_classes != 2):
        test_precision_data  = [[name, prec] for (name, prec) in zip(class_names, t_precision)]
        test_precision_table = wandb.Table(data=test_precision_data, columns=["class_name", "precision"])
    else:
        test_precision_table = wandb.Table()

    ## Recall Table
    if(n_classes != 2):
        test_recall_data  = [[name, prec] for (name, prec) in zip(class_names, t_recall)]
        test_recall_table = wandb.Table(data=test_recall_data, columns=["class_name", "recall"])
    else:
        test_recall_table = wandb.Table()
    
    test_confusion_matrix, saved_confmatrix = get_confusion_matrix(test_preds, test_labels, n_classes, class_names)

    wandb.log({

              "N": N,

              "test loss" : test_loss, 
            #   "test accuracy" : wandb.plot.bar(test_acc_table, "class_name" , "accuracy", title="Test Per Class Accuracy"),
              "test f1" : t_f1,
              "test precision" : t_precision,
              "test recall" : t_recall,
              "test specificity" : t_specificity,
      
              "test class accuracies": wandb.plot.bar(test_acc_table, "class_name" , "accuracy", title="Test Per Class Accuracy"),
              "test class f1"        : wandb.plot.bar(test_f1_table, "class_name" , "f1", title="Test Per Class F1"),
              "test class precision" : wandb.plot.bar(test_precision_table, "class_name" , "precision", title="Test Per Class Precision"),
              "test class recall"    : wandb.plot.bar(test_recall_table, "class_name" , "recall", title="Test Per Class Recall"),

              'test AUROC' : t_auroc,
              
              
              "test_confusion_matrix" : test_confusion_matrix
                })
    # print()

    time_elapsed = time.time() - since
    print('testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Test Acc: {:4f}'.format(test_acc))

    return saved_confmatrix, outs