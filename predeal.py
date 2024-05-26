import os
import numpy as np
from PIL import Image
import matplotlib.pyplot

image_lst=os.listdir("salt")

pic_n=image_lst[0]

image=Image.open(f"salt/{pic_n}").convert("RGB")

