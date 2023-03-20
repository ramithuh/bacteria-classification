from __future__ import print_function, division

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from torchmetrics.classification import Accuracy
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go

import sys
sys.path.append('../../')

from modules.helpers import *
from modules.datasets import *
from modules.train_utils import train_model
from modules.dataloaders import *
from modules.test_utils import test_model

import wandb

path = "/n/home12/ramith/FYP/bacteria-classification/results/ARP - Resnet 181645133220.7868116/latest_model_epoch-8.pth"
saved = torch.load(path)
cfg   = saved['cfg']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_c = cfg['n_classes']

class_names = [x for x in range(0, n_c)]

if(n_c == 21):
    class_names = ['Acinetobacter','B subtilis','E. coli K12','S. aureus','E. coli (CCUG17620)','E. coli (NCTC13441)','E. coli (A2-39)','K. pneumoniae (A2-23)','S. aureus (CCUG35600)','E. coli (101)','E. coli (102)','E. coli (104)','K. pneumoniae (210)','K. pneumoniae (211)','K. pneumoniae (212)','K. pneumoniae (240)','Acinetobacter K12-21','Acinetobacter K48-42','Acinetobacter K55-13','Acinetobacter K57-06','Acinetobacter K71-71']
elif(n_c == 5):
    class_names = ['Acinetobacter', 'B. subtilis', 'E. coli', 'K. pneumoniae', 'S. aureus']


model_ft = models.resnet18(pretrained=cfg['pretrained_resnet'])

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, n_c)

model_ft = model_ft.to(device)

model = model_ft
model.load_state_dict(saved['state_dict']);
model.eval();

data_dir = '/n/holyscratch01/wadduwage_lab/ramith/bacteria_processed'



for N in [63]:
    dataloaders = {}

    _, _, _, _ =  get_bacteria_dataloaders(cfg['img_size'], N , 10, label_type = cfg['label_type'], balanced_mode = False, expand_channels = cfg['expand_channels'])

    dataset_sizes = {'test':0}

    for i in range(0, 21):
        print("=====")
        dataloaders[str(i)], count =  get_bacteria_eval_dataloaders(cfg['img_size'], N , 10, label_type = cfg['label_type'] ,expand_channels = cfg['expand_channels'], isolate_class = i)

        dataset_sizes['test'] += count['test']
        
    wandb.login(key = os.environ.get('WANDB_API_KEY_Bac'))

    wandb.init(project="ARP - Classifier", name = f"Blind Test Evaluation () => {N}", config = cfg,  entity="ramith")
    
    criterion = nn.CrossEntropyLoss()
    
    
    from modules.test_utils import test_model_in_groups
    
    conf, _ = test_model_in_groups(model_ft, [dataloaders, dataset_sizes, class_names, N] , criterion, n_classes = cfg['n_classes'] , device = device, cfg = cfg)
    
    print(conf)
    
    wandb.finish()