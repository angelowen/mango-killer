import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import torchvision.models as models

##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = './C1-P1_Train'


if __name__ == '__main__':
	model16 = models.vgg16(pretrained=True)
	model19 = models.vgg19(pretrained=True)
	print(model16)
	print(model19)

