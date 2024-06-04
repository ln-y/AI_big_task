import os, shutil
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-eps',type=float,help="the eps")
args=parser.parse_args()

noise_cate=["gauss","salt"]
tar_dir=os.path.join("..", "noise_test")  # 修改目标目录路径
noise_id=0
for cate in noise_cate:
    cate_dir=os.path.join("..", f"{cate}_eps={args.eps}")  # 修改噪声类别目录路径
    for fi in os.listdir(cate_dir):
        fi_newname=f"{fi[0]}_{noise_id}.jpg"
        noise_id+=1
        shutil.move(os.path.join(cate_dir, fi), os.path.join(tar_dir, fi_newname))
    shutil.rmtree(cate_dir)
