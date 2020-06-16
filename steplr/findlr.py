import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy,math

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 0
DATASET_ROOT = './C1-P1_Train'

def find_lr(trn_loader,train_set,init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    criterion = nn.CrossEntropyLoss()
    model = VGG16(num_classes=train_set.num_classes)
    model = model.cuda(CUDA_DEVICES)
    model.train()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs.cuda(CUDA_DEVICES)), Variable(labels.cuda(CUDA_DEVICES))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses
        
def train():
    data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
    log_lrs,losses=find_lr(data_loader,train_set)
    print("log_lrs = ",log_lrs)
    print("losses = ",losses)
        
if __name__ == '__main__':
    train()
