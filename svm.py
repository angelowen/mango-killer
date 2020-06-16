from dataset import IMAGE_Dataset
from torch.autograd import Variable
from pathlib import Path
import numpy as np
import torchvision.models as models
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 1
DATASET_ROOT_train = './C1-P1_Train'
DATASET_ROOT_test = './C1-P1_Dev'



def test(model):
	data_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
							 0.229, 0.224, 0.225])
	])
	test_set = IMAGE_Dataset(Path(DATASET_ROOT_test), data_transform)
	data_loader = DataLoader(
		dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
	classes = [_dir.name for _dir in Path(DATASET_ROOT_test).glob('*')]

	model.eval()

	total_correct = 0
	total = 0
	class_correct = list(0. for i in enumerate(classes))
	class_total = list(0. for i in enumerate(classes))
	with torch.no_grad():
		for inputs, labels in data_loader:
			inputs=inputs[:,0,:,:]
			inputs = inputs.reshape(-1, 224 * 224)

			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			# totoal
			total += labels.size(0)
			total_correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()
			# batch size
			for i in range(labels.size(0)):
				label =labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	print('Accuracy on the ALL test images: %d %%'
		  % (100 * total_correct / total))

	for i, c in enumerate(sorted(classes)):
		print('Accuracy of %5s : %2d %%' % (
		c, 100 * class_correct[i] / class_total[i]))
	return (total_correct / total)

def train():
	data_transform = transforms.Compose([

		transforms.Resize((224,224)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3), 
		transforms.RandomRotation(30),
		transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	])
	train_set = IMAGE_Dataset(Path(DATASET_ROOT_train), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)
	model = nn.Linear(224 * 224, 3).to(CUDA_DEVICES)
	model = model.cuda(CUDA_DEVICES)
	model.train()
	
	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 200
	criterion = hinge_loss
	
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
	lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)	

	for epoch in range(num_epochs):

		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0


		for i, (inputs, labels) in enumerate(data_loader):
			inputs=inputs[:,0,:,:]
		#	print("shape:  ",inputs.shape)
			inputs = inputs.reshape(-1, 224 * 224)
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))
		#	print(inputs.size())
			optimizer.zero_grad()
			
			outputs = model(inputs)
		#	print(outputs.size())
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			lr_schduler.step()
			training_loss += loss.item() * inputs.size(0)
		#	print(preds.size(),labels.size())
		#	exit(0)
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

		test_acc=test(model)

		if test_acc > best_acc:
			best_acc = test_acc
			best_model_params = copy.deepcopy(model.state_dict())
		if epoch==100 :
			model.load_state_dict(best_model_params)
			torch.save(model, f'model-{100-best_acc:.02f}-best_train_acc.pth')

	model.load_state_dict(best_model_params)
	torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T
    outputs = outputs[range(num_labels)]
    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss

if __name__ == '__main__':    
	train()
