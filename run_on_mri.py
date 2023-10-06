import argparse
import os
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import imageio
import sys
import time
from datetime import datetime
import monai
from PIL import Image
from tqdm import tqdm

import wandb
from monai.utils import PytorchPadMode, BlendMode
from sklearn.utils import shuffle
from torch.optim import lr_scheduler, rmsprop, Adam

import torch
import numpy as np
from models import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='wbMRI cancer detection')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet2D(1).to(device)
model.load_state_dict(torch.load('/home/abhishekmoturu/Documents/wbmri_cancer_project/31.pt', map_location=device))
model.eval()
inferer = monai.inferers.SlidingWindowInferer(roi_size=(128, 128), mode=BlendMode.GAUSSIAN, overlap=0.5)

with torch.no_grad():
    for i in tqdm(range(1, 51)):
        os.mkdir('/home/abhishekmoturu/Documents/labeling_tool/masks_final')
        os.mkdir('/home/abhishekmoturu/Documents/labeling_tool/masks_final/volume_' + str(i))
        for j in range(len(os.listdir('/home/abhishekmoturu/Documents/labeling_tool/volumes/volume_' + str(i)))):
            image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(cv2.imread('/home/abhishekmoturu/Documents/labeling_tool/volumes/volume_' + str(i) + '/' + str(j) + '.png')[:,:,0]), 0), 0)/255.0
            pred = inferer(inputs=image.to(device), network=model)
            im = Image.fromarray(np.array(torch.softmax(pred, dim=1).argmax(dim=1).cpu().detach()[0]).astype('uint8'))
            im.save('/home/abhishekmoturu/Documents/labeling_tool/masks_final/volume_' + str(i) + '/' + str(j) + '.png')
