### 环境配置
依赖项已在`requirements.txt`列出，使用下述命令安装
```bash
pip install -r requirements.txt
```

### 数据集与权重
由于数据集和权重较大没有上传，请注意数据集的文件夹应该为在此目录的`train`(训练集),`val`(交叉验证集),`test`(测试集)

原所给的图片只有`train`和`val`文件夹，将`val`文件作为`test`文件，运行`split_val.py`即可从`train`中分配图片到`val`

