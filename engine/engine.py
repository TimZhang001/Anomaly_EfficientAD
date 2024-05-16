import os
import sys
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
import tqdm
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append('/home/zhangss/Tim.Zhang/ADetection/Anomaly_EfficientAD')
from models.efficicentADNet import EfficientADNet
from engine.data_loader import get_AD_dataset, load_infinite

class Engine(object):
    def __init__(self, config, mode = 'train'):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode         = mode
        self.config       = config
        self.channel_size = config['Model']['channel_size']
        self.input_size   = config['Model']['input_size']
        self.fmap_size    = (self.input_size,self.input_size)
        self.ckpt_dir     = config['ckpt_dir']
        self.category     = config['category']

        # set seed
        self._set_seed(config['seed'])
        
        # build dataloader
        self._init_dataloader(config)

        # build model
        self.model  = EfficientADNet(config)

    def _init_dataloader(self, config):

        self.teacher_mean = None
        self.teacher_std  = None
        self.batch_size   = config['Model']['batch_size']
        self.print_freq   = config['print_freq']
        self.data_transforms = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                                transforms.ToTensor(),
                                                ])
        self.gt_transforms = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                                transforms.ToTensor()])
        

        #We obtain an image P ∈ R 3×256×256 from ImageNet by choosing a random image,
        # resizing it to 512 × 512,
        # converting it to gray scale with a probability of 0.3 and cropping the center 256 × 256 pixels
        teacher_input   = config['Datasets']['imagenet']['teacher_input']
        grayscale_ratio = config['Datasets']['imagenet']['grayscale_ratio']
        self.imagenet_transforms = transforms.Compose([transforms.Resize((teacher_input, teacher_input)),        
                                                    transforms.RandomGrayscale(p=grayscale_ratio),            
                                                    transforms.CenterCrop((self.input_size,self.input_size)), 
                                                    transforms.ToTensor(),])
                
    def _load_datasets(self):
        # --------------------------------------------------------------------
        normalize_dataset = get_AD_dataset(type=self.config['Datasets']['train']['type'],  
                                           root=self.config['Datasets']['train']['root'],
                                           transform=self.data_transforms,
                                           gt_transform=self.gt_transforms,
                                           phase='train',
                                           category=self.category,
                                           split_ratio=1)
        normalize_dataloader = DataLoader(normalize_dataset,batch_size=1,shuffle=True,num_workers=4, pin_memory=True)
        
        # --------------------------------------------------------------------
        dataset = get_AD_dataset(type=self.config['Datasets']['train']['type'],
                                 root=self.config['Datasets']['train']['root'],
                                 transform=self.data_transforms,
                                 gt_transform=self.gt_transforms,
                                 phase='train',
                                 category=self.category,
                                 split_ratio=0.8)
        train_dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,num_workers=4, pin_memory=True)
        train_dataloader = load_infinite(train_dataloader)
        print('load train dataset:length:{}'.format(len(dataset)))
        
        # --------------------------------------------------------------------
        quantile_dataset = get_AD_dataset(type=self.config['Datasets']['train']['type'],
                                          root=self.config['Datasets']['train']['root'],
                                          transform=self.data_transforms,
                                          gt_transform=self.gt_transforms,
                                          phase='eval',
                                          category=self.category,
                                          split_ratio=0.8)
        quantile_dataloader = DataLoader(quantile_dataset,batch_size=1,shuffle=True,num_workers=4, pin_memory=True)
        
        # --------------------------------------------------------------------
        imagenet_path = os.path.join(self.config['Datasets']['imagenet']['root'])
        if not os.path.exists(imagenet_path):
            imagenet_iterator = None
        else:
            imagenet = get_AD_dataset(type='ImageNet',
                                    root=self.config['Datasets']['imagenet']['root'],
                                    transform=self.imagenet_transforms,)
            imagenet_loader   = DataLoader(imagenet,batch_size=1,shuffle=True,num_workers=4, pin_memory=True)
            imagenet_iterator = load_infinite(imagenet_loader)
        
        # --------------------------------------------------------------------
        eval_dataset = get_AD_dataset(type=self.config['Datasets']['train']['type'],
                                      root=self.config['Datasets']['train']['root'],
                                      transform=self.data_transforms,
                                      gt_transform=self.gt_transforms,
                                      phase='test',
                                      category=self.category)
        eval_dataloader = DataLoader(eval_dataset,batch_size=1,shuffle=True)
        
        # --------------------------------------------------------------------
        return normalize_dataloader, train_dataloader, imagenet_iterator, quantile_dataloader, eval_dataloader

    def _set_seed(self,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_optimizer(self, iterations):
        optimizer  = optim.Adam(list(self.model.st_model.parameters()) +
                                list(self.model.ae_model.parameters()),lr=0.0001,weight_decay=0.00001)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * iterations), gamma=0.1)

        return optimizer, scheduler
    
    def _cal_teacher_mean_std(self, dataloader):
        
        # calculate teacher mean and std
        num         = 0
        input_data  = torch.randn(1,3,self.input_size,self.input_size).cuda()
        temp_tensor = self.model.te_model(input_data)
        x           = torch.zeros((500,self.channel_size,*temp_tensor.shape[2:]))
        for item in tqdm.tqdm(dataloader):
            if num>=500:
                break
            ldist = item['image'].cuda()
            y     = self.model.te_model(ldist).detach().cpu()
            yb    = y.shape[0]
            x[num:num+yb,:,:,:] = y[:,:,:,:]
            num += yb
        self.teacher_mean = x[:num,:,:,:].mean(dim=(0,2,3),keepdim=True).cuda()
        self.teacher_std  = x[:num,:,:,:].std(dim=(0,2,3),keepdim=True).cuda()

        # save teacher mean and std
        teacher_std_ckpt  = "{}/{}_teacher_mean_std.pth".format(self.config['ckpt_dir'], self.config['category'])
        torch.save({'mean':self.teacher_mean,'std':self.teacher_std},teacher_std_ckpt)
        
    def train(self,iterations=70000):

        # dataset
        normalize_dl,train_dl,imagenet_iterator,quantile_dl,eval_dl = self._load_datasets()
        
        # cal teacher model normalization
        self._cal_teacher_mean_std(normalize_dl)
        
        # optimizer
        optimizer, scheduler = self._init_optimizer(iterations)     
          
        best_auroc = 0
        best_loss  = 100
        print('start train iter:',iterations)
        for i_batch in range(iterations):
            sample_batched = next(train_dl)
            image          = sample_batched['image'].cuda()
            
            self.model.st_model.train()
            self.model.ae_model.train()
            loss_st    = self.model.loss_st(image, imagenet_iterator,   self.model.te_model, self.model.st_model)
            LAE,LSTAE  = self.model.loss_ae(image, self.model.te_model, self.model.st_model, self.model.ae_model)
            loss_total = loss_st + LAE + LSTAE

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            if i_batch % self.print_freq == 0:
                print("label:{},batch:{}/{},loss_total:{:.4f},loss_st:{:.4f},loss_ae:{:.4f},loss_stae:{:.4f}".format(
                    self.category,i_batch,iterations,loss_total.item(),loss_st.item(),LAE.item(),LSTAE.item()))

                # 计算map参数
                self.qa_st,self.qb_st,self.qa_ae,self.qb_ae = self.map_norm_quantiles(quantile_dl)
                quantiles = {'qa_st':self.qa_st,'qb_st':self.qb_st,'qa_ae':self.qa_ae,'qb_ae':self.qb_ae}

                # save model and map 参数
                torch.save(quantiles, '{}/{}_quantiles_last.pth'.format(self.ckpt_dir,self.category))
                
                if loss_total < best_loss:    
                    auroc = self.eval(eval_dl)
                    if auroc > best_auroc:
                        best_loss  = loss_total
                        best_auroc = auroc
                        print('saving model in {} at auroc:{:.4f}'.format(self.ckpt_dir, auroc))                 
                        torch.save(self.model, '{}/{}_best.pth'.format(self.ckpt_dir,self.category))  

            if best_auroc > 0.995:
                print('best auroc > 0.995, break')
                break  
        print('train done')
                                          
    def eval(self, eval_dataloader):
        scores = []
        gts    = []
        for sample_batched in tqdm.tqdm(eval_dataloader):
            gts.append(sample_batched['label'].item())
            combined_map, image_score = self.infer_single(sample_batched)
            scores.append(image_score.item())
        gtnp    = np.array(gts)
        scorenp = np.array(scores)
        auroc   = roc_auc_score(gtnp,scorenp)
        return auroc

    def infer_single(self, sample_batched):
        img = sample_batched['image']
        img = img.cuda()
        with torch.no_grad():
            combined_map, image_score = self.model(img)
        return combined_map, image_score
    
    def infer(self):
        pass

    def map_norm_quantiles(self,dataloader):
        maps_st, maps_ae = [],[]
        self.model.st_model.eval()
        self.model.ae_model.eval()
        self.model.te_model.eval()
        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched = sample_batched['image'].cuda()
            map_st, map_ae = self.model.predict(sample_batched)
            maps_st.append(map_st)
            maps_ae.append(map_ae)
        
        # --------------------------------
        maps_st = torch.cat(maps_st)
        maps_ae = torch.cat(maps_ae)
        qa_st   = torch.quantile(maps_st, q=0.9)
        qb_st   = torch.quantile(maps_st, q=0.995)
        qa_ae   = torch.quantile(maps_ae, q=0.9)
        qb_ae   = torch.quantile(maps_ae, q=0.995)
        
        return qa_st,qb_st,qa_ae,qb_ae

