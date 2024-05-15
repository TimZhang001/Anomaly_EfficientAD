import torch.nn as nn
import torch
import torch.nn.functional as F
from wide_resnet import imagenet_norm_batch

class PDN_S(nn.Module):

    def __init__(self, last_kernel_size=384,with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        self.with_bn = with_bn
        self.conv1   = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2   = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4   = nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        return x
    
class PDN_M(nn.Module):

    def __init__(self, last_kernel_size=384,with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 256 3 ReLU
        # AvgPool-1 2×2 2×2 256 1 -
        # Conv-2 1×1 4×4 512 3 ReLU
        # AvgPool-2 2×2 2×2 512 1 -
        # Conv-3 1×1 1×1 512 0 ReLU
        # Conv-4 1×1 3×3 512 1 ReLU
        # Conv-5 1×1 4×4 384 0 ReLU
        # Conv-6 1×1 1×1 384 0 -
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(last_kernel_size, last_kernel_size, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(256)
            self.bn2 = nn.BatchNorm2d(512)
            self.bn3 = nn.BatchNorm2d(512)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(last_kernel_size)
            self.bn6 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x) if self.with_bn else x
        return x


class Teacher(nn.Module):
    def __init__(self,size, with_bn=False, channel_size=384, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size =='M':
            self.pdn = PDN_M(last_kernel_size=channel_size,with_bn=with_bn)
        elif size =='S':
            self.pdn = PDN_S(last_kernel_size=channel_size,with_bn=with_bn)
        # self.pdn.apply(weights_init)

    def forward(self, x):
        #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = imagenet_norm_batch(x) 
        x = self.pdn(x)
        return x
    

class Student(nn.Module):
    def __init__(self,size,with_bn=False,channel_size=768, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size =='M':
            #The student network has the same architecture,but 768 kernels instead of 384 in the Conv-5 and Conv-6 layers.
            self.pdn = PDN_M(last_kernel_size=channel_size,with_bn=with_bn) 
        elif size =='S':
            #The student network has the same architecture, but 768 kernels instead of 384 in the Conv-4 layer
            self.pdn = PDN_S(last_kernel_size=channel_size,with_bn=with_bn) 
        # self.pdn.apply(weights_init)

    def forward(self, x):
        #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = imagenet_norm_batch(x) 
        pdn_out = self.pdn(x)
        return pdn_out