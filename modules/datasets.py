import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import glob
import cv2
import torchvision
import matplotlib.pyplot as plt

### Species level mapping
# 0 => Acinetobacter
# 1 => B subtilis
# 2 => E. coli
# 3 => K. pneumoniae
# 4 => S. aureus
#More info => https://ruhsoft-my.sharepoint.com/:p:/g/personal/im_ramith_fyi/EYMDb528EVlClCp2y8nIM8oB9LBZ-lbqEiCXwcAZHX7wew?e=lAROoR

species_mapping_dict  = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 4,
    9: 2,
    10: 2,
    11: 2,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
}
class bacteria_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get the bacteria dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/bacteria_np', type_= 'train', transform= None, label_type = "class", balance_data = False, expand_channels = False):
        self.transform= transform
        self.label_type = label_type
        self.type_ = type_
        self.expand_channels = expand_channels

        all_dirs = sorted(glob.glob(f'{data_dir}/{type_}/*/*'), key= lambda x: int(x.split('/')[-1][:-4]))

        print(f"Dataset type {type_} label type: {label_type}", end = " -> ")

        ### Extract portion of all files for each class
        dirs = {}

        for i,x in enumerate(all_dirs):
            # data  = np.load(x, allow_pickle=True)[1]
            # class_ = self.__getclass_(data, self.label_type)

            class_ = int(x.split('/')[-2])
            
            if(class_ in dirs.keys()):
                dirs[class_].append(x)
            else:
                dirs[class_] = [x]

        img_dirs_filtered = []

        ## Get the class with minimum count
        min_class_count = 1000000000
        if(balance_data):
            for i in range(0,21):
                count = len(dirs[i])
                if(count < min_class_count):
                    min_class_count = count
            print(" - Min class count: ", min_class_count)

        for i in range(0,21): # iterate through all classes
            if(balance_data):
                count = min_class_count
            else:
                count = len(dirs[i])

            img_dirs_filtered.append(dirs[i][:int(count)]) 
                
        self.img_dirs = [item for sublist in img_dirs_filtered for item in sublist] # flatten list

        print(f"Loaded {len(self.img_dirs)} images")

        
    def __len__(self):
        return len(self.img_dirs)

    def __getclass_(self, meta_data, label_type):
        if(label_type == 'class'):
            return meta_data[0]
        elif(label_type == 'wild_type'):
            return meta_data[1]
        elif(label_type == 'gram_strain'):
            return meta_data[2]
        elif(label_type == 'species'):
            return species_mapping_dict[meta_data[0]] #map class to species
        else:
            raise Exception("Invalid label type")
        
    def __getitem__(self, idx): 
        data  = np.load(self.img_dirs[idx], allow_pickle=True)
        image = data[0]
        

        label = self.__getclass_(data[1], self.label_type)

        if self.transform:
            image = self.transform(image)
        
        if(self.expand_channels):
            image = image.expand(3, image.shape[1], image.shape[1])
        
        return image, label





