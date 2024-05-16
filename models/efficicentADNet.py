import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import sys
import os

sys.path.append('/home/zhangss/Tim.Zhang/ADetection/Anomaly_EfficientAD')

from models.pdn import Student, Teacher
from models.autoencoder import AutoEncoder
from torchvision import transforms

class EfficientADNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config       = config
        self.ckpt_dir     = config['ckpt_dir']
        self.category     = config['category']
        self.model_size   = config['Model']['model_size']
        self.with_bn      = config['Model'].get('with_bn',False)
        self.with_bn      = str(self.with_bn).lower()=='true'
        self.channel_size = config['Model']['channel_size']

        self.input_size = config['Model']['input_size']
        self.score_in_mid_size=int(0.9*self.input_size)


        self.teacher_mean = None
        self.teacher_std  = None
        self.qa_st        = None
        self.qb_st        = None
        self.qa_ae        = None
        self.qb_ae        = None

        self.init_models()
             
    def init_models(self):
        self.st_model = Student(self.model_size, self.with_bn)
        self.st_model = self.st_model.cuda()
        # self.student.apply(weights_init)
        
        self.te_model = Teacher(self.model_size, self.with_bn)
        self.load_pretrain_teacher()

        self.ae_model = AutoEncoder(is_bn=self.with_bn)
        self.ae_model = self.ae_model.cuda()
        # self.ae.apply(weights_init)

    def load_teacher_mean_std(self):
        teacher_std_ckpt  = "{}/{}_teacher_mean_std.pth".format(self.ckpt_dir, self.category)
        assert os.path.exists(teacher_std_ckpt), 'teacher mean and std file not found'
        mean_std          = torch.load(teacher_std_ckpt)
        self.teacher_mean = mean_std['mean']
        self.teacher_std  = mean_std['std']
        print('load channel mean and std from {}'.format(teacher_std_ckpt))

    def load_quantile(self):
        quantile_ckpt = '{}/{}_quantiles_last.pth'.format(self.ckpt_dir,self.category)
        assert os.path.exists(quantile_ckpt), 'quantile file not found'
        quantiles     = torch.load(quantile_ckpt)
        self.qa_st     = quantiles['qa_st']
        self.qb_st     = quantiles['qb_st']
        self.qa_ae     = quantiles['qa_ae']
        self.qb_ae     = quantiles['qb_ae']

    def load_pretrain_teacher(self):
        ckpt_path = '{}/best_teacher.pth'.format(self.ckpt_dir)
        self.te_model.load_state_dict(torch.load(ckpt_path))
        self.te_model = self.te_model.cuda()
        self.te_model.eval()
        for parameters in self.te_model.parameters():
            parameters.requires_grad = False
        print('load teacher model from {}'.format(ckpt_path))


    def predict(self, image):
        teacher_output = self.te_model(image)
        teacher_output = (teacher_output-self.teacher_mean)/self.teacher_std
        student_output = self.st_model(image)
        encoder_output = self.ae_model(image)
    
        distance_st = torch.pow(teacher_output-student_output[:, :self.channel_size, :, :],2)
        distance_ae = torch.pow(encoder_output-student_output[:, self.channel_size:, :, :],2)
        map_st      = torch.mean(distance_st, dim=1, keepdim=True)
        map_ae      = torch.mean(distance_ae, dim=1, keepdim=True)
        
        return map_st, map_ae
        
    def forward(self, image):
        teacher_output = self.te_model(image)
        teacher_output = (teacher_output-self.teacher_mean)/self.teacher_std
        student_output = self.st_model(image)
        encoder_output = self.ae_model(image)
        
        #3: Split the student output into Y ST ∈ R 384×64×64 and Y STAE ∈ R 384×64×64 as above
        distance_st = torch.pow(teacher_output-student_output[:, :self.channel_size, :, :],2)
        distance_ae = torch.pow(encoder_output-student_output[:, self.channel_size:, :, :],2)
        map_st      = torch.mean(distance_st, dim=1, keepdim=True)
        map_ae      = torch.mean(distance_ae, dim=1, keepdim=True)
        
        if self.qb_st is None or self.qb_ae is None or self.qb_ae is None or self.qb_st is None:
            self.load_quantile()
     
        map_st = (0.1*(map_st - self.qa_st)) / (self.qb_st - self.qa_st)
        map_ae = (0.1*(map_ae - self.qa_ae)) / (self.qb_ae - self.qa_ae)

        combined_map = 0.5*map_st + 0.5*map_ae
        combined_map = F.interpolate(combined_map, size=(self.input_size,self.input_size), mode='bilinear')
        idx_start    = (self.input_size - self.score_in_mid_size)//2
        image_score  = torch.max(combined_map[:,:,idx_start:idx_start+self.score_in_mid_size,
                                              idx_start:idx_start+self.score_in_mid_size])
        return combined_map, image_score

    # ----------------------------------------------------------------- #
    def choose_random_aug_image(self,image):
        aug_index   = random.choice([1,2,3])
        coefficient = random.uniform(0.8,1.2)
        if aug_index == 1:
            img_aug = transforms.functional.adjust_brightness(image,coefficient)
        elif aug_index == 2:
            img_aug = transforms.functional.adjust_contrast(image,coefficient)
        elif aug_index == 3:
            img_aug = transforms.functional.adjust_saturation(image,coefficient)
        return img_aug

    def loss_st(self,image,imagenet_iterator,teacher:Teacher,student:Student):
        
        if self.teacher_mean is None or self.teacher_std is None:
            self.load_teacher_mean_std()
              
        with torch.no_grad():
            teacher_out = teacher(image)
            teacher_out = (teacher_out-self.teacher_mean)/self.teacher_std
        
        student_out = student(image)
        student_out = student_out[:, :self.channel_size, :, :]
        distance_st = torch.pow(teacher_out-student_out,2)
        dhard       = torch.quantile(distance_st[:8,:,:,:],0.999)
        Lhard       = torch.mean(distance_st[distance_st>=dhard])
        
        if imagenet_iterator is not None:
            image_p      = next(imagenet_iterator)
            imagenet_out = student(image_p[0].cuda())
            loss_penalty = torch.mean(torch.pow(imagenet_out[:, :self.channel_size, :, :],2))
            loss_st      = Lhard + loss_penalty
        else:
            loss_st      = Lhard
        return loss_st
    
    def loss_ae(self,image,teacher:Teacher,student:Student,autoencoder:AutoEncoder):
        
        if self.teacher_mean is None or self.teacher_std is None:
            self.load_teacher_mean_std()
        
        aug_img = self.choose_random_aug_image(image=image)
        aug_img = aug_img.cuda()
        with torch.no_grad():
            teacher_out = teacher(aug_img)
            teacher_out = (teacher_out-self.teacher_mean)/self.teacher_std
        encoder_out = autoencoder(aug_img)
        student_out = student(aug_img)
        student_out = student_out[:, self.channel_size:, :, :]
        
        dist_ae   = torch.pow(teacher_out-encoder_out,2)
        dist_stae = torch.pow(student_out-encoder_out,2)
        LAE       = torch.mean(dist_ae)
        LSTAE     = torch.mean(dist_stae)
        return LAE, LSTAE
    