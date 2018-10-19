#coding:utf-8
# 将图片裁剪
# 高度-4 上下各减2，变为56
# 宽度-4，左右各减2，变为176


from PIL import Image
import os


TRAIN_IMG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'pic','train')
filenames = os.listdir(TRAIN_IMG_PATH)

for filename in filenames:
    _file = TRAIN_IMG_PATH +os.sep + filename
    img = Image.open(_file)
    w,h = img.size
    new_img = img.crop([2,2,w-2,h-2])
    new_img.save(_file)
