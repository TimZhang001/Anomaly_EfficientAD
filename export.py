import argparse
import yaml
import os
import sys
import torch

sys.path.append('/home/zhangss/Tim.Zhang/ADetection/Anomaly_EfficientAD')

from models.efficicentADNet import EfficientADNet

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/mvtec_train.yaml')
    parser.add_argument('--category',   type=str, default='')
    parser.add_argument('--root_dir',   type=str, default='')
    parser.add_argument('--ckpt_dir',   type=str, default='')    
    parser.add_argument('--iterations', type=int, default=None)    
    args = parser.parse_args()
    return args

def parse_args(args):
    # if args.config:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.category!="":
        config['category'] = args.category
    if args.root_dir!="":
        config['train']['root'] = args.root_dir
        config['eval']['root']  = args.root_dir
    if args.ckpt_dir!="":
        config['ckpt_dir'] = args.ckpt_dir
    if args.iterations:
        config['train']['iterations'] = args.iterations
    return config 

if __name__ == '__main__':

    args   = get_arguments()
    config = parse_args(args)
    
    # model and load best checkpoint
    ckpt_path = os.path.join(config['ckpt_dir'], '{}_best.pth'.format(config['category']))
    model     = EfficientADNet(config=config)
    model     = torch.load(ckpt_path)
    model.eval()

    with torch.no_grad():
        # 使用 TorchScript 跟踪模型
        input_tensor = torch.rand(1, 3, 256, 256)
        input_tensor = input_tensor.cuda()
        traced_model = torch.jit.trace(model, input_tensor)

        # 保存 TorchScript 模型
        ckpt_save_path = os.path.join(config['ckpt_dir'], '{}_best_traced.pt'.format(config['category']))
        traced_model.save(ckpt_save_path)


