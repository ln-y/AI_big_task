# this file is used to generate train, val , test dataset for contrast model

import os,shutil
import random



file_path_lst=[]

src_dir_lst=["../test","../val","../train"]
dst_dir_lst=["test","val","train"]
splot_ratio_lst=[0, 0.2 ,0.4, 1]

for diri in dst_dir_lst:
    if os.path.exists(diri):
        shutil.rmtree(diri)
    os.makedirs(diri)

pic_id=0

for src_dir in src_dir_lst:
    for r,d,f in os.walk(src_dir):
        for fi in f:
            if ("aigc" not in fi) and ("noise" not in fi):
                new_name=f"{fi[0]}_{pic_id}.jpg"
                pic_id+=1
                file_path_lst.append((f"{r}/{fi}",new_name))

print(f"{pic_id=}\n{len(file_path_lst)=}")

random.shuffle(file_path_lst)

for ind,dst_dir in enumerate(dst_dir_lst):
    for i in range(int(len(file_path_lst)*splot_ratio_lst[ind]),int(len(file_path_lst)*splot_ratio_lst[ind+1])):
        shutil.copy(file_path_lst[i][0],f"{dst_dir}/{file_path_lst[i][1]}")