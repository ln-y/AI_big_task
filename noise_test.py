# generate noise test data to path: noise/
import os, shutil
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-eps',type=float,help="the eps")
args=parser.parse_args()

noise_cate=["gauss","salt"]
tar_dir="noise_test"
noise_id=0
for cate in noise_cate:
    cate_dir=f"{cate}_eps={args.eps}"
    for fi in os.listdir(cate_dir):
        fi_newname=f"{fi[0]}_{noise_id}.jpg"
        noise_id+=1
        shutil.move(f"{cate_dir}/{fi}",f"{tar_dir}/{fi_newname}")
    shutil.rmtree(cate_dir)