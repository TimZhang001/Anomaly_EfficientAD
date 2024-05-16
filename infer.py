import argparse
import yaml
import os
import sys
import torch

sys.path.append('/home/zhangss/Tim.Zhang/ADetection/Anomaly_EfficientAD')

from engine.engine import Engine

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
    best_ckpt_path = os.path.join(config['ckpt_dir'], '{}_best.pth'.format(config['category']))
    engine         = Engine(config=config)
    engine.model   = torch.load(best_ckpt_path)
    engine.model.eval()

    # infer
    engine.infer()