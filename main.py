import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from KNN import KNN 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


