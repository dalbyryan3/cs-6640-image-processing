from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io, transform
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
import random


class RegistrationDatasetLoader(Dataset):
    def __init__(self, csv_file='TrainingDatasetClassification.csv', root_dir_input='./', root_dir_ref='./',transform=None):
        self.name_csv = pd.read_csv(csv_file)
        self.root_dir_input = root_dir_input
        self.root_dir_ref = root_dir_ref
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.name_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_img_name   = os.path.join(self.root_dir_ref,self.name_csv.iloc[idx, 0])
        input_img_name = os.path.join(self.root_dir_input,self.name_csv.iloc[idx, 1])
        ref_image      = io.imread(ref_img_name)
        input_image    = io.imread(input_img_name)
        prob= random.uniform(0, 1) 
        if(prob<0.5):
            sample = {'ref_image': ref_image, 'inputImage': input_image,'x-shift':self.name_csv.iloc[idx, 2],'y-shift':self.name_csv.iloc[idx, 3]}
        else:
            sample = {'ref_image': input_image, 'inputImage': ref_image,'x-shift':-1*self.name_csv.iloc[idx, 2],'y-shift':-1*self.name_csv.iloc[idx, 3]}

        if self.transform:
            sample['ref_image'] = self.transform(sample['ref_image'])
            sample['inputImage'] = self.transform(sample['inputImage'])
        return sample


