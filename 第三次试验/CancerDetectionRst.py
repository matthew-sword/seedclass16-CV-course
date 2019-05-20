import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from tqdm import tqdm_notebook
import random
from sklearn.model_selection import train_test_split
from fastai import *
from fastai.vision import *
from torchvision.models import *
   

ORIGINAL_SIZE = 96      # 剪裁前图像原始大小
CROP_SIZE = 90          # 剪裁后图像大小
RANDOM_ROTATION = 3     # 随机旋转角度
RANDOM_SHIFT = 2        # 中心随机平移
RANDOM_BRIGHTNESS = 7   # 随机亮度
RANDOM_CONTRAST = 5     # 随机对比度
RANDOM_90_DEG_TURN = 1  # 0 或 1= 随机做或者右旋转90度
BATCH_SIZE = 128                    # 指定batch大小
ARCH = densenet169                  # 指定cnn网络模型
MODEL_PATH = str(ARCH).split()[1]   # 模型保存位置
sz = CROP_SIZE                      # 剪裁后图片大小


# 图片与处理函数
def readCroppedImage(path, augmentations = True):
    # 用于图像增强
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations
    
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    
    if(not augmentations):
        return rgb_img / 255
    
    # 随机旋转
    rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)
    if(RANDOM_90_DEG_TURN == 1):
        rotation += random.randint(-1,1) * 90
    M = cv2.getRotationMatrix2D((48,48),rotation,1)   # the center point is the rotation anchor
    rgb_img = cv2.warpAffine(rgb_img,M,(96,96))
    
    # 随机平移x,y-shift
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    
    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255
    
    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if(flip_hor):
        rgb_img = rgb_img[:, ::-1]
    if(flip_ver):
        rgb_img = rgb_img[::-1, :]
        
    # 随机亮度
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = rgb_img + br
    
    # 随机对比度
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = rgb_img * cr
    
    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    
    return rgb_img

# 调用图片预处理函数的子类
class MyImageItemList(ImageList):
    def open(self, fn:PathOrStr)->Image:
        # fn 图像路径
        img = readCroppedImage(fn)  # 对图像进行处理
        # 图像矩阵需要先进行转换，转换为tensor 再传递给fastai
        return vision.Image(px=pil2tensor(img, np.float32))
    


#############################数据集处理#############################
train_path = './train/'
test_path = './test/'
data = pd.read_csv('./train_labels.csv')

#print('data type', type(data).__name__)
labelMsg = data['label'].value_counts()
print('label Msg :\n', labelMsg)

train_df = data.set_index('id')
train_names = train_df.index.values     # 图片名称
train_labels = np.asarray(train_df['label'].values)     # 图片标签

# 训练集切分，用于交叉验证
tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)

# 为模型创建训练数据帧 train_names = 220025
train_dict = {'name': train_path + train_names, 'label': train_labels}
df_train = pd.DataFrame(data=train_dict)


# 创建测试数据帧
test_names = []
for f in os.listdir(test_path):
    test_names.append(test_path + f)
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])


# 创建fastai数据集
imgDataBunch = (MyImageItemList.from_df(path='./', df=df_train, suffix='.tif')
        
        .split_by_idx(val_idx)

        .label_from_df(cols='label')

        .add_test(MyImageItemList.from_df(path='./', df=df_test))

        .transform(tfms=[[],[]], size=sz)
       
        .databunch(bs=BATCH_SIZE)

        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])

       )

print('imgDataBunch type', type(imgDataBunch))

#############################建立模型#############################


# ps = drop out rate
def getLearner():
    return cnn_learner(imgDataBunch, ARCH, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

learner = getLearner()

print('build model')
#############################参数调优#############################

# 只训练头部其它部分冻结
max_lr = 2e-2
wd = 1e-4
# CLR策略
learner.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd)

# 交叉验证
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')

# 保存模型
learner.save(MODEL_PATH + '_stage1')

# 加载模型
learner.load(MODEL_PATH + '_stage1')

# 解冻模型继续训练
learner.unfreeze()

# 减小学习率
learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5,4e-4))


# 再次交叉验证
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')

# 保存模型
learner.save(MODEL_PATH + '_stage2')


# 进行预测
print('start predict')

preds,y, loss = learner.get_preds(with_loss=True)
# 打印准确率
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))
