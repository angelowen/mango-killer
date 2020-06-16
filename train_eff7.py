import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import numpy as np
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 1
DATASET_ROOT_train = './C1-P1_Train'
DATASET_ROOT_test = './C1-P1_Dev'

def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
	cycle = np.floor(1 + iteration/(2  * stepsize))
	x = np.abs(iteration/stepsize - 2 * cycle + 1)
	lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
	return lr

def get_dynamic_momentum(iteration, stepsize, base_lr, max_lr):
	cycle = np.floor(1 + iteration/(2  * stepsize))
	x = np.abs(iteration/stepsize - 2 * cycle + 1)
	mm = base_lr - (max_lr - base_lr) * np.maximum(0, (1-x))
	return mm

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
		transforms.CenterCrop((200,200)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3), 
		transforms.RandomRotation(30),
		transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		#transforms.Lambda(lambda img: img * 2.0 - 1.0)

	])
	train_set = IMAGE_Dataset(Path(DATASET_ROOT_train), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)
	model = EfficientNet.from_pretrained('efficientnet-b7')
	model = model.cuda(CUDA_DEVICES)
	model.train()
	
	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 200
	criterion = nn.CrossEntropyLoss()
	
	stepsize = 20
	base_lr = 0.001
	max_lr = 0.01
	base_mm=0.8
	max_mm=0.99
	

	for epoch in range(num_epochs):
		#newlr = get_triangular_lr(epoch,stepsize,base_lr,max_lr)
		#mm=get_dynamic_momentum(epoch,stepsize,base_mm,max_mm)
		optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0


		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()
			
			outputs = model(inputs)

			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
		#	print(training_loss)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

		test_acc=test(model)

		if test_acc > best_acc:
			best_acc = test_acc
			best_model_params = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model_params)
	torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')


if __name__ == '__main__':
	train()
