#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
import time
from tqdm import tqdm
from common.common import get_autoencoder, get_pdn_small, get_pdn_medium, ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
import cv2


mvtecAD     = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
mvtecLocoAD = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
MuraAD      = ['MonoLocalAD']

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',    default='mura_ad', choices=['mvtec_ad', 'mvtec_loco', 'mura_ad'])
    parser.add_argument('-s', '--subdataset', default=['all'], help='One of 15 sub-datasets of Mvtec AD or 5 sub-datasets of Mvtec LOCO or all')
    parser.add_argument('-o', '--output_dir', default='output')
    parser.add_argument('-m', '--model_size', default='small', choices=['small', 'medium'])
    parser.add_argument('-w', '--weights',    default='ckpts/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path', default='none',
                        help='Set to "none" to disable ImageNet pretraining penalty. Or see README.md to download ImageNet and set to ImageNet path')
    
    parser.add_argument('-a', '--ad_path',      default='./data/ADetection/', help='AD dataset')
    parser.add_argument('-t', '--train_steps',  default=10000,     type=int, ) # 70000
    return parser.parse_args()

# constants
seed   = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size   = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

# ---------------------
def run_train(config, dataset_path, pretrain_penalty=False):
    # 当前的日期
    datestr = time.strftime("%Y%m%d", time.localtime())

    # create output dir
    train_output_dir = os.path.join(config.output_dir, datestr, 'trainings',    config.dataset, config.subdataset)
    test_output_dir  = os.path.join(config.output_dir, datestr, 'anomaly_maps', config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # load data
    full_train_set = ImageFolderWithoutTarget(os.path.join(dataset_path, config.subdataset, 'train'), 
                                              transform=transforms.Lambda(train_transform))
    test_set       = ImageFolderWithPath(os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad' or config.dataset == 'mura_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size      = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng             = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,[train_size, validation_size], rng)
    elif config.dataset == 'mvtec_loco':
        train_set      = full_train_set
        validation_set = ImageFolderWithoutTarget(os.path.join(dataset_path, config.subdataset, 'validation'), 
                                                  transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')


    train_loader          = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader     = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        penalty_set    = ImageFolderWithoutTarget(config.imagenet_train_path, transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj  = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard      = torch.quantile(distance_st, q=0.999)
        loss_hard   = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae   = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae    = torch.mean(distance_ae)
        loss_stae  = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description("Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 500 == 0:
            torch.save(teacher, os.path.join(train_output_dir,'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

        # 70000 -> 10000   10000->2000
        if iteration % 1000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(validation_loader=validation_loader, teacher=teacher,
                                                                           student=student, autoencoder=autoencoder,
                                                                           teacher_mean=teacher_mean, teacher_std=teacher_std,
                                                                           desc='Intermediate map normalization')
            auc = test(test_set=test_set, teacher=teacher, student=student,
                       autoencoder=autoencoder, teacher_mean=teacher_mean,
                       teacher_std=teacher_std, q_st_start=q_st_start,
                       q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                       test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()

            if auc > 99.99:
                print('Early stopping because auc is 100%')
                break

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(validation_loader=validation_loader, 
                                                                   teacher=teacher, student=student,
                                                                   autoencoder=autoencoder, teacher_mean=teacher_mean,
                                                                   teacher_std=teacher_std, desc='Final map normalization')
    auc = test(test_set=test_set, teacher=teacher, student=student,
               autoencoder=autoencoder, teacher_mean=teacher_mean,
               teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
               q_ae_start=q_ae_start, q_ae_end=q_ae_end,
               test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))

# ---------------------
def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        subdatasets  = mvtecAD if config.subdataset[0] in ['all'] else config.subdataset
        dataset_path = os.path.join(config.ad_path, "mvtec")
    elif config.dataset == 'mvtec_loco':
        subdatasets  = mvtecLocoAD if config.subdataset[0] in ['all']  else config.subdataset
        dataset_path = os.path.join(config.ad_path, "mvtecloco")
    elif config.dataset == 'mura_ad':
        subdatasets  = MuraAD if config.subdataset[0] in ['all']  else config.subdataset
        dataset_path = os.path.join(config.ad_path, "MuraAD")
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = False if config.imagenet_train_path == 'none' else True
    for subdataset in subdatasets:
        assert subdataset in mvtecAD + mvtecLocoAD + MuraAD, 'Unknown subdataset'
        config.subdataset = subdataset
        print('Training on', config.subdataset)
        run_train(config, dataset_path, pretrain_penalty)
    
# ---------------------
def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true  = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        #orig_width  = image.width
        #orig_height = image.height
        image       = default_transform(image)
        orig_width  = image.shape[2]
        orig_height = image.shape[1]
        image       = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(image=image, teacher=teacher, student=student,
                                               autoencoder=autoencoder, teacher_mean=teacher_mean,
                                               teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
                                               q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = np.transpose(map_combined[0, 0].cpu().numpy())
        map_save     = cv2.applyColorMap(np.uint8((map_combined + 1) * 128), cv2.COLORMAP_JET)

        map_st       = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
        map_st       = torch.nn.functional.interpolate(map_st, (orig_height, orig_width), mode='bilinear')
        map_st       = np.transpose(map_st[0, 0].cpu().numpy())
        map_st       = cv2.applyColorMap(np.uint8((map_st + 1) * 128), cv2.COLORMAP_JET)

        map_ae       = torch.nn.functional.pad(map_ae, (4, 4, 4, 4))
        map_ae       = torch.nn.functional.interpolate(map_ae, (orig_height, orig_width), mode='bilinear')
        map_ae       = np.transpose(map_ae[0, 0].cpu().numpy())
        map_ae       = cv2.applyColorMap(np.uint8((map_ae + 1) * 128), cv2.COLORMAP_JET)

        image        = image[0].cpu().numpy()
        mean         = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))  
        std          = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))  
        image        = np.transpose(np.uint8((image * std + mean) *255))

        image_save   = np.concatenate([image, map_save, map_st, map_ae], axis=1) 

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            os.makedirs(os.path.join(test_output_dir, defect_class), exist_ok=True)
            img_nm = os.path.split(path)[1].split('.')[0]
            file   = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, image_save)

        y_true_image  = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output     = teacher(image)
    teacher_output     = (teacher_output - teacher_mean) / teacher_std
    student_output     = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:])**2, dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []

    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(image=image, teacher=teacher, student=student,
                                               autoencoder=autoencoder, teacher_mean=teacher_mean, 
                                               teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    
    # --------------------------------
    maps_st    = torch.cat(maps_st)
    maps_ae    = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end   = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end   = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
