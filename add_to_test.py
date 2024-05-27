import os,shutil
import random
from typing import List

test_id=0
val_id=0
train_id=0
test_ratio, val_ratio= 0.7,0.1

noise_test_ratio, noise_val_ratio = 0 ,0.2

if os.path.exists('aigc_test'):
    shutil.rmtree("aigc_test")

if os.path.exists('noise_test'):
    shutil.rmtree("noise_test")
    
if os.path.exists('all_test'):
    shutil.rmtree("all_test")
    
os.makedirs("aigc_test")
os.makedirs("all_test")
os.makedirs("noise_test")

for fi in os.listdir("test"):
    shutil.copy(f"test/{fi}",f"all_test/{fi}")

for fi in os.listdir("val"):
    if '_aigc.jpg' in fi or '_noise.jpg' in fi:
        os.remove(f"val/{fi}")

for fi in os.listdir("train"):
    if '_aigc.jpg' in fi or '_noise.jpg' in fi:
        os.remove(f"train/{fi}")

# noise
for r,d,f in os.walk('noise'):
    fi_lst:List[str]=[]
    for fi in f:
        if fi.endswith('jpg'):
            fi_lst.append(f'{r}/{fi}')
    print(len(fi_lst))
    test_num=len(fi_lst)*noise_test_ratio
    val_num=len(fi_lst)*noise_val_ratio
    random.shuffle(fi_lst)
    for i in range(int(test_num)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"all_test/{cate}_{test_id}_noise.jpg"
        new_name1=f"noise_test/{cate}_{test_id}_noise.jpg"
        test_id+=1
        shutil.copy(fi_lst[i],new_name)
        shutil.copy(fi_lst[i],new_name1)
    for i in range(int(test_num),int(test_num+val_num)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"val/{cate}_{val_id}_noise.jpg"
        val_id+=1
        shutil.copy(fi_lst[i],new_name)
    for i in range(int(test_num+val_num),len(fi_lst)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"train/{cate}_{train_id}_noise.jpg"
        train_id+=1
        shutil.copy(fi_lst[i],new_name)
print(test_id,val_id,train_id)

test_id=0
val_id=0
train_id=0
# aigc
for r,d,f in os.walk('images'):
    fi_lst:List[str]=[]
    for fi in f:
        if fi.endswith('jpg'):
            fi_lst.append(f'{r}/{fi}')
    print(len(fi_lst))
    test_num=len(fi_lst)*test_ratio
    val_num=len(fi_lst)*val_ratio
    random.shuffle(fi_lst)
    for i in range(int(test_num)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"all_test/{cate}_{test_id}_aigc.jpg"
        new_name1=f"aigc_test/{cate}_{test_id}_aigc.jpg"
        test_id+=1
        shutil.copy(fi_lst[i],new_name)
        shutil.copy(fi_lst[i],new_name1)
    for i in range(int(test_num),int(test_num+val_num)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"val/{cate}_{val_id}_aigc.jpg"
        val_id+=1
        shutil.copy(fi_lst[i],new_name)
    for i in range(int(test_num+val_num),len(fi_lst)):
        fi=fi_lst[i]
        ind=fi.rfind('/')
        cate=fi[ind+1]
        new_name=f"train/{cate}_{train_id}_aigc.jpg"
        train_id+=1
        shutil.copy(fi_lst[i],new_name)

print(test_id,val_id,train_id)



