from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
import random
import torch.optim as optim

class NoiseDatsetLoader(Dataset):
    def __init__(self, csv_file='TrainingDataSet.csv', root_dir_noisy='TrainingDataSet', root_dir_ref='./',transform=None):
        self.name_csv = pd.read_csv(csv_file)
        self.root_dir_noisy = root_dir_noisy
        self.root_dir_ref = root_dir_ref
        self.transform = transform

    def __len__(self):
        return len(self.name_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_img_name = os.path.join(self.root_dir_ref,self.name_csv.iloc[idx, 0])
        noisy_img_name = os.path.join(self.root_dir_noisy,self.name_csv.iloc[idx, 2])
        ref_image    = np.array(io.imread(ref_img_name))
        noisy_image    = np.array(io.imread(noisy_img_name))
        ref_image    = np.expand_dims(ref_image, axis=0) 
        noisy_image    = np.expand_dims(noisy_image, axis=0) 
        #ref_image = ref_image/256
        #noisy_image = noisy_image/256
        sample = {'image': ref_image, 'NoisyImage': noisy_image}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['NoisyImage'] = self.transform(sample['NoisyImage'])
        return sample





