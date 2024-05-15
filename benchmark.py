#!/usr/bin/python
# -*- coding: utf-8 -*-
# ---------------------- 进行网络的运行时间的测试 ----------------------
from time import time
import numpy as np
import torch.cuda
from torch import nn


# image_out = floor(floor(image_in - 3) /2 -3) / 2 - 2 - 3
def get_pdn(out=384):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=1),       # -3
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),                                                        # 1/2
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1),     # -3
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),                                                        # 1/2
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),     # -2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out, kernel_size=4, stride=1),     # -3
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out, out_channels=out, kernel_size=1, stride=1)
    )

def get_ae():
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),  # 1/2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), # 1/2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), # 1/2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), # 1/2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), # 1/2
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8), # 从8*8 ->1*1
        
        # decoder
        nn.Upsample(3, mode='bilinear'),                                                # 1 -> 3
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 3 + 1
        nn.ReLU(inplace=True),   #
        nn.Upsample(8, mode='bilinear'),                                                # 4 -> 8
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 8 + 1
        nn.ReLU(inplace=True),
        nn.Upsample(15, mode='bilinear'),                                               # 9 -> 15
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 15 + 1
        nn.ReLU(inplace=True),
        nn.Upsample(32, mode='bilinear'),                                               # 16 -> 32
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 32 + 1
        nn.ReLU(inplace=True),
        nn.Upsample(63, mode='bilinear'),                                               # 33 -> 63
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 63 + 1
        nn.ReLU(inplace=True),
        nn.Upsample(127, mode='bilinear'),                                              # 64 -> 127
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2), # 127 + 1
        nn.ReLU(inplace=True),
        nn.Upsample(56, mode='bilinear'),                                               # 128 -> 56   #120
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3, stride=1, padding=1) 
    )


gpu = torch.cuda.is_available()

autoencoder = get_ae()
teacher = get_pdn(384)
student = get_pdn(768)

autoencoder = autoencoder.eval()
teacher = teacher.eval()
student = student.eval()

if gpu:
    autoencoder.half().cuda()
    teacher.half().cuda()
    student.half().cuda()

quant_mult = torch.e
quant_add = torch.pi
with torch.no_grad():
    times = []
    for rep in range(2000):
        image = torch.randn(1, 3, 256, 256, dtype=torch.float16 if gpu else torch.float32)
        start = time()
        if gpu:
            image = image.cuda()

        t = teacher(image)
        s = student(image)
        
        st_map = torch.mean((t - s[:, :384]) ** 2, dim=1)
        ae     = autoencoder(image)
        ae_map = torch.mean((ae - s[:, 384:]) ** 2, dim=1)
        st_map = st_map * quant_mult + quant_add
        ae_map = ae_map * quant_mult + quant_add
        result_map = st_map + ae_map
        result_on_cpu = result_map.cpu().numpy()
        timed = time() - start
        times.append(timed)
print(np.mean(times[-1000:]))

