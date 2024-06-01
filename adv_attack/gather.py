# this file used to gather all adv_sample to one dir

import os, shutil
pic_id=0

tar_dir="../contrast_adv"

if os.path.exists(tar_dir):
    shutil.rmtree(tar_dir)
os.makedirs(tar_dir)

for r,d,f in os.walk("."):
    for fi in f:
        if fi.endswith('.png'):
            shutil.copy(f"{r}/{fi}",f"{tar_dir}/{fi[0]}_{pic_id}.png")
            pic_id+=1