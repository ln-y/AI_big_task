import os, shutil
import random

#using this code to split `test`
f_lst=[]
for r,d,f in os.walk("train"):
    for fi in f:
        f_lst.append(fi)
print(len(f_lst),f_lst[:10])
split_ratio=0.2
split_num=split_ratio*len(f_lst)
random.shuffle(f_lst)
tar_lst=f_lst[:int(split_num)]
print(tar_lst[:10])
for fi in tar_lst:
    shutil.move(f"train/{fi}",f"val/{fi}")


f1_lst=[]
for r,d,f in os.walk("val"):
    for fi in f:
        f1_lst.append(fi)
print(len(f1_lst),f1_lst[:10])