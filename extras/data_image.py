from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
from medmnist import ChestMNIST
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

data = np.load('/Users/pnans/.medmnist/chestmnist.npz')
plt.imshow(data['train_images'][10])
plt.savefig("hehe2.png")
