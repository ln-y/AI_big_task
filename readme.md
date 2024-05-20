### 环境配置
由于版本与大作业指导手册同步，采用了较老的版本，环境配置较为特殊，请遵循以下操作：

**linux**
```bash
#进入到对应的python环境
source runme.sh
```

**windows**
```bash
pip install -r requirements.txt
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install numpy==1.19.0
```

### 数据集与权重
由于数据集和权重较大没有上传，请注意数据集的文件夹应该为在此目录的`train`(训练集),`val`(交叉验证集),`test`(测试集)

原所给的图片只有`train`和`val`文件夹，将`val`文件作为`test`文件，运行`split_val.py`即可从`train`中分配图片到`val`

