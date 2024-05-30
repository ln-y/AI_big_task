# used to generate aigc train, val, test file

import os, shutil
import random

test_num=2
val_ratio=0.2

dir_lst=[diri.name for diri in os.scandir(os.getcwd()) if diri.is_dir()]
print(len(dir_lst))
random.shuffle(dir_lst)

# del all aigc pic in val, train, aigc_test
def remove_pic(prefix:str,path:str):
    for fi in os.scandir(path):
        if prefix in fi.name:
            os.remove(f"{path}/{fi.name}")

remove_pic("_aigc.jpg","../val")
remove_pic("_aigc.jpg","../train")

if os.path.exists("../aigc_test"):
    shutil.rmtree("../aigc_test")

os.makedirs("../aigc_test")

train_id=0
val_id=0
test_id=0

for dir in dir_lst[:2]:
    for fi in os.listdir(dir):
        cate=fi[0]
        new_name=f"{cate}_{test_id}_aigc.jpg"
        test_id+=1
        shutil.copy(f"{dir}/{fi}",f"../aigc_test/{new_name}")

for dir in dir_lst[2:]:
    lst=os.listdir(dir)

    for fi in lst[:int(val_ratio*len(lst))]:
        cate=fi[0]
        new_name=f"{cate}_{val_id}_aigc.jpg"
        val_id+=1
        shutil.copy(f"{dir}/{fi}",f"../val/{new_name}")
    
    for fi in lst[int(val_ratio*len(lst)):]:
        cate=fi[0]
        new_name=f"{cate}_{train_id}_aigc.jpg"
        train_id+=1
        shutil.copy(f"{dir}/{fi}",f"../train/{new_name}")

print(val_id,train_id,test_id)