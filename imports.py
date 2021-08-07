import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torchvision
import cv2
import os
import random
import wandb
import face_recognition
from tqdm import tqdm
from torch.nn import *
from torch.optim import *
from PIL import Image

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
epochs = 50
IMG_SIZE = 84
device = "cuda"
criterion = CrossEntropyLoss()
optimizer = Adam
batch_size = 32
PROJECT_NAME = "Age-Prediction"
lr = 0.001
