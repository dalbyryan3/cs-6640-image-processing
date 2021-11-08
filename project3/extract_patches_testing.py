import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as transforms 
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os.path
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import csv
from skimage import filters

def writeCSV(csv_file_name, Trainingdictlist):
    csv_columns = ['ref_image','input_image','x-shift','y-shift']
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in Trainingdictlist:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def main( image_paths, sampled_ref, sampled_input):   
    if not os.path.exists(sampled_ref):
        os.makedirs(sampled_ref)
    if not os.path.exists(sampled_input):
        os.makedirs(sampled_input)
    image_names = pd.read_csv('testing_dataset.csv')
    TrainingDict = [] 
    for index in range(0,1):
        print(index)
        image_name=image_names.iloc[index,0]
        image_name_with_path = image_paths+'/'+image_name+'.jpg'
        print(image_name_with_path)
        image = cv2.imread(image_name_with_path)
         
        M,N,C = image.shape
    
        sample_size=256
        k=0
        for i in range(0,500):
            SampleX = np.random.randint(0,M-256)
            SampleY = np.random.randint(0,M-256)
            for j in range(0,5):
                 randomShiftx = np.random.randint(0,64)
                 randomShifty = np.random.randint(0,64)
                 image_cut       = image[SampleX:SampleX+sample_size,SampleY:SampleY+sample_size,:]
                 image_cut_shift = image[SampleX+randomShiftx:SampleX+sample_size+randomShiftx,SampleY+randomShifty:SampleY+sample_size+randomShifty,:]
                 M1, N1,_ = image_cut.shape
                 M2, N2,_ = image_cut_shift.shape
                 if(np.mean(filters.sobel(image_cut))>0.06 and M1 == 256 and N1==256 and M2 == 256 and N2==256 ):
                     imageName             = sampled_ref    + image_name + str(k) +'.png'
                     image_cut_shift_name  = sampled_input  + image_name + str(k) +'.png' 
                     cv2.imwrite(imageName,image_cut)
                     cv2.imwrite(image_cut_shift_name,image_cut_shift)
                     TrainingDict.append({'ref_image':imageName,'input_image':image_cut_shift_name,'x-shift':randomShiftx, 'y-shift':randomShifty})
                     k=k+1
    writeCSV('TestingDatasetClassification.csv',TrainingDict)

if __name__ == "__main__":
    folder_data = "./WSI/"
    preprocess_output_image ="TranslationDataset/Test/input/"
    preprocess_output_ref   ="TranslationDataset/Test/reference/"
    main(folder_data,  preprocess_output_ref, preprocess_output_image)
