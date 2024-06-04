# this file used to convert a *.ckpt to *.pth file
# usage: python convert.py -ckpt <path of *.ckpt> -pth <path of *.pth>

import argparse

import torch

from model import ViolenceClassifier

parser=argparse.ArgumentParser()
parser.add_argument("-ckpt",type=str,help="(src)path of *.ckpt")
parser.add_argument("-pth",type=str,help="(dst)path of *.pth")
args=parser.parse_args()

model=ViolenceClassifier.load_from_checkpoint(args.ckpt)
torch.save(model.model.state_dict(),args.pth)
