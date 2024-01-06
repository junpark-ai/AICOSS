import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
import torch.nn.functional as F
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from datetime import datetime
import wandb
from models_ import * # TODO

from torch.nn import BCEWithLogitsLoss
from utils_ import *
                                
from randaugment import RandAugment
import argparse

warnings.filterwarnings(action='ignore') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())


CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10, # Your Epochs,
    'LR':3e-4, # Your Learning Rate,
    'BATCH_SIZE':8, # Your Batch Size,
    'WEIGHT_DECAY': 1e-5,
    'MIN_LR': 1e-5,
    'SEED':41,

    'model_name': 'cvt_q2l',  # TODO
    'path': './DATA/',  # TODO
    'loss_name': 'PartialSelectiveLoss',  # TODO
}
## Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--num-classes', default=60)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--simulate_partial_type', type=str, default=None, help="options are fpc or rps")
parser.add_argument('--simulate_partial_param', type=float, default=1000)
parser.add_argument('--partial_loss_mode', type=str, default="negative")
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--gamma_pos', type=float, default=0)
parser.add_argument('--gamma_neg', type=float, default=1)
parser.add_argument('--gamma_unann', type=float, default=2)
parser.add_argument('--alpha_pos', type=float, default=1)
parser.add_argument('--alpha_neg', type=float, default=1)
parser.add_argument('--alpha_unann', type=float, default=1)
parser.add_argument('--likelihood_topk', type=int, default=5)
parser.add_argument('--prior_path', type=str, default=None)
parser.add_argument('--prior_threshold', type=float, default=0.05)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--print-freq', '-p', default=128, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--path_dest', type=str, default="./outputs")
parser.add_argument('--debug_mode', type=str, default="hyperml")

args = parser.parse_args()






## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        label = torch.tensor(self.dataframe.iloc[idx][list(self.dataframe.columns)[2:]], dtype=torch.int8) if 'airplane' in self.dataframe.columns else 0
        
        return img, label


    
    
# 데이터 로드
path = CFG['path']
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')

def rewrite(df):
    df['img_path'] = path + df['img_path']
    return df

train = rewrite(train)
test = rewrite(test)
train_data, val_data = train_test_split(train, test_size=0.1, random_state=78, shuffle=True) # stratify=train[list(train.columns)[2:]]


## Train
def val(model, criterion, val_loader, epoch):
    
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, leave=True)
    predict_np = []
    truth_np = []
    with torch.no_grad():
        for imgs, label in val_loop:
            imgs, label = imgs.float().to(device), label.float().to(device)
            # Forward & Loss
            predicted_label = model(imgs)
            loss = criterion(predicted_label.squeeze(1), label)
            
            APs = []
            label_np = label.cpu().detach().numpy()
            pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

            predict_np += pred_np.tolist()
            truth_np += label_np.tolist()
            
            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())
            
        total_val_map = average_precision_score(np.array(truth_np), np.array(predict_np))
        
    total_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1} val loss: {total_val_loss:.4f} val map = {total_val_map:.4f}\n")
    
    return total_val_loss, total_val_map
    
def train(model, criterion, optimizer, train_loader, scheduler=None, val_loader=None):
    wandb.init(entity='aicoss-rcvuos', project="CvT", name= CFG['model_name'] + '-' +CFG['loss_name'], notes=str(torch.cuda.get_device_name())+' x '+str(1))
    # 학습
    prior = ComputePrior(train_loader.dataset.__getitem__(0)[1])
    for epoch in range(CFG['EPOCHS']):
        model.train()
        total_loss, map = 0, 0
        loop = tqdm(train_loader, leave=True)
        for imgs, label in loop:
            imgs, label = imgs.float().to(device), label.float().to(device)
            
            # Forward & Loss
            predicted_label = model(imgs)

            loss = criterion(predicted_label.squeeze(1), label)
            prior.update(predicted_label)
            APs = []
            label_np = label.cpu().detach().numpy()
            pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

            for i in range(predicted_label.shape[1]):
                APs.append(average_precision_score(label_np[:, i], pred_np[:, i]))

            ap = np.mean(APs)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            
            
            total_loss += loss.item()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item(), ap=ap.item())
            map += ap.item()

        print(f"Epoch {epoch + 1} train loss: {total_loss / len(train_loader):.4f} train map = {map / len(train_loader):.4f}")


        if scheduler != None:
            scheduler.step()
            
        if val_loader != None:
            val_loss, val_map = val(model, criterion, val_loader, epoch)
        else:
            val_loss, val_map = 0, 0
        
        wandb.log({"Epoch": epoch + 1,
                   "lr": optimizer.param_groups[0]["lr"],
                   "train loss": total_loss / len(train_loader),
                   "train map": map / len(train_loader),
                   "val loss": val_loss,
                   "val map": val_map})         
            

train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    # transforms.RandomHorizontalFlip(0.5),
    CutoutPIL(cutout_factor=0.7),
    RandAugment(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    Lighting(0.1), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델, 손실함수, 옵티마이저
model = globals()[CFG['model_name']]()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)


if CFG['loss_name'] == 'PartialSelectiveLoss':
    criterion = globals()[CFG['loss_name']](args)
else:
    criterion = globals()[CFG['loss_name']]

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])  # TODO
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG['EPOCHS'], eta_min=CFG['MIN_LR'])  # TODO

train_dataset = CustomDataset(train_data, train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=1)

val_dataset = CustomDataset(val_data, transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)


train(model, criterion, optimizer, train_loader, scheduler, val_loader)


# 현재 시간 불러옴
now = datetime.now()
now_time = now.strftime("%m%d_%H%M")

torch.save(model.state_dict(), f'{CFG["model_name"]}-{now_time}.pt')


## Inference & Submit
test_dataset = CustomDataset(test, transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)

model.eval()
predicted_label_list = []

test_loop = tqdm(test_loader, leave=True)
with torch.no_grad():
    for imgs, label in test_loop:
        imgs, label = imgs.float().to(device), label.float().to(device)
        # Forward & Loss
        predicted_label_list += nn.Sigmoid()(model(imgs)).tolist()
        

# 결과 저장

predicted = pd.DataFrame(predicted_label_list, columns=list(sample_submission.columns)[1:])
result = pd.concat([sample_submission['img_id'], predicted], axis=1)

result.to_csv(f'{CFG["model_name"]}-{now_time}.csv', index=False)
print("Inference completed and results saved to csv file.")