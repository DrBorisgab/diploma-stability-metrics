
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

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]),  dtype=np.float32)
        mask = np.array(Image.open(self.mask_paths[idx]),  dtype=np.float32)[:,:,0] # !костыль - хорошо бы все пересохранить

        #image = self.crop_samples(img)
        #mask = self.crop_samples(mask)

        # if mask.sum() != 0:
        #     print(idx)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask[mask > 0] = 1
        mask = mask.float().unsqueeze(0)

        return {'img': image, 'mask': mask}
        
class ModelPl(pl.LightningModule):
    def __init__(self, model, criterion, lr, apls=False):
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

class SimplePredictor(nn.Module):
    def __init__(self, model, device='gpu', 
                 sample_size=(512,512),
                 #sample_size=(1024,1024), 
                 pad_size=32):
        super(SimplePredictor, self).__init__()
        self.sample_size = sample_size
        self.device = device
        self.model = model
        self.pad_size = pad_size

    def add_border(self, img):
        old_size = (img.shape[1], img.shape[2])

        new_size = ((int(old_size[0] / self.sample_size[0]) + 1)*self.sample_size[0], 
                    (int(old_size[1] / self.sample_size[1]) + 1)*self.sample_size[1])
            
        new_img = torch.zeros((img.shape[0], new_size[0], new_size[1]))

        self.add_x = [int((new_size[0]-old_size[0])/2), int((new_size[0]-old_size[0])/2)]
        self.add_y = [int((new_size[1]-old_size[1])/2), int((new_size[1]-old_size[1])/2)]

        if new_size[0]-np.sum(self.add_x) != img.shape[1]:
            self.add_x[1] = self.add_x[1] + 1

        if new_size[1]-np.sum(self.add_y) != img.shape[2]:
            self.add_y[1] = self.add_y[1] + 1

        new_img[:, self.add_x[0]:new_size[0]-self.add_x[1], 
                   self.add_y[0]:new_size[1]-self.add_y[1]] = img

        return new_img

    def predict(self, img):
        
        self.x_samples = img.shape[1]//self.sample_size[0]
        self.y_samples = img.shape[2]//self.sample_size[1]

        ss = self.sample_size
        mask = torch.zeros((img.shape[1], img.shape[2]))
        self.model.eval()
        with torch.no_grad():
          for i in range(self.x_samples):
              for j in range(self.y_samples):

                  pad = nn.ZeroPad2d(self.pad_size)
                  sample = img[:,i*ss[0]:(i+1)*ss[0], j*ss[1]:(j+1)*ss[1]]
                  sample = sample.unsqueeze(dim=0).to(device)
                  sample = pad(sample)
                  pred = self.model(sample.float())[0,0].sigmoid().detach().cpu()
                  mask[i*ss[0]:(i+1)*ss[0], j*ss[1]:(j+1)*ss[1]] = pred[self.pad_size:pred.shape[0]-self.pad_size, 
                                                                        self.pad_size:pred.shape[1]-self.pad_size]

        return mask

    def restore_size(self, img):

        img = img[self.add_x[0]:img.shape[0]-self.add_x[1],
                  self.add_y[0]:img.shape[1]-self.add_y[1]]

        return img

    def forward(self, img):

        img = self.add_border(img)
        mask = self.predict(img)
        mask = self.restore_size(mask)

        return  mask

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    pl.seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    imgaug.random.seed(seed)

    print(f"Random seed set as {seed}")

def get_test_dataloader(path, augs, batch_size, num_workers=0):
    valid_dataset = FromDiscDataset(path, transform = augs)
    return DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


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

        # if self.train_mode == 'random':
        #     self.train_dataset = RandomDataset(self.data_dir+'train/', transform=self.train_aug, num_samples=self.num_samples)

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
