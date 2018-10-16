from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


PATH = "/Users/Ezhil/Documents/data_science/dogscats/"
sz=224
cuba=torch.cuda.is_available()
print (cuba)
torch=torch.backends.cudnn.enabled
print (torch)
dir_check=os.listdir(PATH)
print (dir_check)
dir_check1=os.listdir(f'{PATH}valid')
print(dir_check1)
files = os.listdir(f'{PATH}valid/cats')[:5]
print(files)
img = plt.imread(f'{PATH}valid/cats/{files[0]}')
img1=plt.imshow(img)
print(img1)
shape=img.shape
print(shape)
print(img[0:3,0:3])
shutil.rmtree(f'{PATH}tmp', ignore_errors=True)
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)
