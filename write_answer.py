import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
import pandas as pd


CUDA_DEVICES = 1
DATASET_ROOT = './C1-P1_Test'
PATH_TO_WEIGHTS = './mymodel/model-0.83-best_train_acc.pth'
train_path='./C1-P1_Train'

def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    dataset_root = Path(DATASET_ROOT)
    classes = [_dir.name for _dir in Path(train_path).glob('*')]
    print(classes)
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    sample_submission = pd.read_csv("./test_example.csv")
    submission =sample_submission.copy()
    for i,filename in enumerate(sample_submission['image_id']):
        image =Image.open(Path(DATASET_ROOT).joinpath(filename)).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICES))
        outputs =model(inputs)
        _,preds = torch.max(outputs.data,1)
        submission['label'][i] = classes[preds[0]]
    submission.to_csv("./test_example.csv",index=False)

if __name__ == '__main__':
    test()
