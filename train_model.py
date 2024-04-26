!pip install -r requirements.txt
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import random
import glob
import seaborn as sns
from inference.py import *
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import JaccardLoss
from segmentation_models_pytorch.utils.metrics import IoU
import warnings
import networkx as nx
import os
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")
import pandas as pd
pl.seed_everything(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import rasterio
from aeronet_vector import FeatureCollection, Feature
import shapely
import numpy as np
from matplotlib import pyplot as plt
import os
from dlutils.data import markup_generation, fcutils, angleutils, heightutils, markup_generation
from dlutils.utils import visualization, npfile_utils
import cv2 
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from pathlib import Path

pl.seed_everything(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.253317  , 0.26740879, 0.23025433]
std  = [0.15158384, 0.14880167, 0.14123519]

train_augs = A.Compose([
    A.Flip(p=0.5),
    A.Rotate(border_mode=0, p=0.8),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.RandomBrightness(limit=0.1, p=0.5),
    A.RandomContrast(limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
    A.PadIfNeeded(512,512, border_mode=0),
    A.CenterCrop(512,512, always_apply=True),
    ToTensorV2()
])

val_augs = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0)
])

# optimize get_item
class FromDiscDataset(Dataset):

    def __init__(self, data_dir, sample_size=(512,512), 
                 transform=None, cpAug=False):

        self.transform = transform
        self.sample_size = sample_size
        self.cpAug = cpAug
        #self.images, self.masks = self.create_samples(data_dir)
        self.img_paths = sorted(glob.glob(data_dir+'/images/*.tif'))
        self.mask_paths = sorted(glob.glob(data_dir+'/masks/*.tif'))
 

    def __len__(self):
        return len(self.img_paths)

    def pad_if_needed(self, img):
        old_size = img.size

        x_size = (int(old_size[0] / self.sample_size[0]) + 1)*self.sample_size[0] if old_size[0] % self.sample_size[0] != 0 else old_size[0]
        y_size = (int(old_size[1] / self.sample_size[1]) + 1)*self.sample_size[1] if old_size[1] % self.sample_size[1] != 0 else old_size[1]
        new_size = (x_size, y_size) 

        if new_size == old_size:
            return img
            
        new_img = Image.new(img.mode, new_size, 0)
        new_img.paste(img, (int((new_size[0]-old_size[0])/2),
                            int((new_size[1]-old_size[1])/2)))
            
        return new_img

    def crop_samples(self, img):
        
        img = self.pad_if_needed(img)
        x_samples = img.size[0]//self.sample_size[0]
        y_samples = img.size[1]//self.sample_size[1]

        samples = []
        ss = self.sample_size
        for i in range(x_samples):
            for j in range(y_samples):
                sample = img.crop((i*ss[0], j*ss[0], (i+1)*ss[0], (j+1)*ss[1]))

                # сделали dtype=np.float32 чтобы маски из булевого типа перевести во флоат
                print(sample.shape)
                samples.append(np.array(sample, dtype=np.float32))

        return samples

    def __getitem__(self, idx):

        image = np.array(Image.open(self.img_paths[idx]),  dtype=np.float32)
        mask = np.array(Image.open(self.mask_paths[idx]),  dtype=np.float32)[:,:,0] # !костыль - хорошо бы все пересохранить

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        mask[mask > 0] = 1
        mask = mask.float().unsqueeze(0)
        return {'img': image, 'mask': mask}
    
    
class PlDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_aug=None, val_aug=None,
                 num_samples=1000, batch_size=15, num_workers=4,
                 train_dset_mode='full', use_tpu=False,
                 cpAug=False):
      
        super().__init__()
        assert train_dset_mode in ['random', 'full']
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.train_mode = train_dset_mode
        self.use_tpu = use_tpu
        self.cpAug = cpAug

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if self.train_mode == 'random':
            self.train_dataset = RandomDataset(self.data_dir+'train/', transform=self.train_aug, num_samples=self.num_samples)

        if self.train_mode == 'full':
            self.train_dataset = FromDiscDataset(self.data_dir+'train/', transform=self.train_aug)
            
        self.valid_dataset = FromDiscDataset(self.data_dir+'val/', transform=self.val_aug)
        print('dm loaded')

    def train_dataloader(self):
        train_sampler = None
        if self.use_tpu:
          train_sampler = DistributedSampler(self.train_dataset,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          sampler=train_sampler, 
                          num_workers=self.num_workers, 
                          shuffle=True, 
                          # persistent_workers=True, 
                          pin_memory=True) # add persistent_workers=True
# 
    def val_dataloader(self):
        val_sampler = None
        if self.use_tpu:
          val_sampler = DistributedSampler(self.valid_dataset,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=False)
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers)


class ModelPl(pl.LightningModule):
    def __init__(self, model, criterion, lr=1e-3, apls=False):
        super(ModelPl, self).__init__()

        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.IoU = IoU()
        self.lr = lr
        self.apls=apls

    def forward(self, z):
        return self.model(z)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.5)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        pred = self.model(batch['img'])
        loss = self.criterion(pred.sigmoid(), batch['mask'])
        iou = self.IoU(pred.sigmoid().detach().cpu(), batch['mask'].detach().cpu())

        self.log('train loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train IoU', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):

        pred = self.model(batch['img'])

        loss = self.criterion(pred.sigmoid(), batch['mask'])
        iou = self.IoU(pred.sigmoid().detach().cpu(), batch['mask'].detach().cpu())

        self.log('val loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val IoU', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


CFG = {
    'SAVE_DIR': 'E:/diplom_dataset/samples_new/',
    'run_name': str(datetime.now()),
    'project_name': 'unet_resnet34_2024',
    'epochs': 5, #20
    'sample_size': (512,512),
    'batch_size': 8,
    'num_workers': 0, #2
    # 'num_samples': 1500,
    'lr': 1e-4,#, 1e-4],
    'use_tpu': False,
    }

dm = PlDataModule(data_dir = CFG['SAVE_DIR'],
                  train_aug=train_augs, 
                  val_aug=val_augs,
                  batch_size=CFG['batch_size'], #1
                  num_workers=CFG['num_workers'], 
                  use_tpu=CFG['use_tpu'], 
                  train_dset_mode='full', 
                  cpAug=False)

dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

save_path = CFG['SAVE_DIR']+CFG['run_name']

wandb_logger = pl.loggers.WandbLogger(name=CFG['run_name']+'_'+str(CFG['epochs'])+'_'+str(CFG['batch_size']), 
                                      project=CFG['project_name'])
# callback = pl.callbacks.ModelCheckpoint(monitor='val IoU',save_top_k=1, 
#                                         # dirpath=CFG['SAVE_DIR'],
#                                         #filename=CFG['run_name'],
#                                           mode='max')
model = ModelPl(smp.Unet(encoder_name="resnet34", encoder_weights="imagenet"), JaccardLoss())
trainer = pl.Trainer(#gpus = 1, 
                     #precision = 16,
                     accelerator='gpu',
                     max_epochs = CFG['epochs'], 
                     logger = wandb_logger,
                     enable_progress_bar=True,
                     #progress_bar_refresh_rate = 2,
                     # callbacks = callback,
                     #callbacks = [TQDMProgressBar(refresh_rate=1)],
                     num_sanity_val_steps=0
                     #accumulate_grad_batches=2,
                     )

trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('unet_resnet34/simple_pipline_w_augs_'+ str(CFG['epochs']) + '_'+ str(CFG['batch_size']) + '_'+ str(CFG['num_workers']) + '_'+ str(datetime.now()) +'.ckpt')