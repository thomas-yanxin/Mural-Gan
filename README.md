# Mural-Gan

# 壁画效果预览
以下第一张为原图，第二张为经过Style1风格迁移后的效果图，第三张为经过Style2风格迁移后的效果图。总体来说，效果还是可以的。
![原图](https://ai-studio-static-online.cdn.bcebos.com/7823b0bfca694cff8fd4681facbd5058d308048944ee434eae2160033889ec02)
![Style1_tranfer](https://ai-studio-static-online.cdn.bcebos.com/a9a0eb6e343a4610a46cb6624f297710ded2e3cfe13d48a69f6aad4272232060)
![Style2_tranfer](https://ai-studio-static-online.cdn.bcebos.com/f56e6e24c46941fea5bf96f9bc27028b93ef4704b69b45219f3581014346a288)


# 背景介绍

&emsp;&emsp;壁画，墙壁上的艺术，即人们直接画在墙面上的画。作为建筑物的附属部分，壁画的装饰和美化功能使它成为环境艺术的一个重要方面。**壁画为人类历史上最早的绘画形式之一**。  

&emsp;&emsp;壁画是指绘在建筑物的墙壁或天花板上的图案。它分为粗底壁画、刷底壁画和装贴壁画等。壁画是最古老的绘画形式之一。至今埃及、印度、巴比伦、中国等文明古国保存了不少古代壁画。在意大利文艺复兴时期，壁画创作十分繁荣，产生了许多著名的作品。我国自周代以来，历代宫室乃至墓室都有饰以壁画；随着宗教信仰的兴盛，壁画又广泛应用于寺观、石窟（例如**敦煌莫高窟**、**芮城永乐宫**等）。我国至今仍大量保存着著名的佛教壁画和道教壁画遗迹。这些遗迹有部分已经被列入了**世界文化遗产的保护名录，作为我们古代文明的见证**。  

&emsp;&emsp;我国陕西咸阳秦皇宫壁画残片，距今有2300年。唐代是我国壁画的兴盛时期，那段时期是我国壁画艺术的高峰期，创作出了很多古今闻名的壁画，如敦煌壁画、克孜尔石窟等。到了宋代以后，壁画逐渐衰落。直到1949年后，中国壁画才逐渐得到恢复与发展 。  

&emsp;&emsp;随着现代社会生活多元化、个性化的发展，艺术上出现了更多的以“**表现自我**”为中心的艺术形式，现代壁画与建筑环境艺术为了顺应社会发展的需要，出现了多种多样的艺术形式，为人们创造了一个多元化的文化环境。正壁画艺术伴随着人类文明的发展，具有辉煌而悠久的历史，它是人类追求美的理想、表现内心精神世界的独特艺术形式。随着人们审美观念的变化，城市化进程的不断加快，现代建筑环境对壁画的要求，壁画语言发生了较大的变化，现代壁画无论从表现形式、语言内涵到其社会功能都与传统壁画有着较大差异。  

&emsp;&emsp;现代壁画随着社会的发展也在不断的进步与发展，在互联网没有普及的岁月，人们更青睐于写实的、具象的、内容繁冗丰富的，但随着互联网的诞生与普及，人们受着过量信息的困扰，内心变得疲惫，需要的是放松、明快、简约的东西被再一次的肯定，**壁画领域也随之而改变，不断地推陈出新，来符合人们的需要与发展** 。  

&emsp;&emsp;在建筑业蓬勃发展的今天，大多数建筑承袭现代主义“**少就是多**”的观念，外墙装饰均是清一色的玻璃幕墙或单色瓷砖，面目冷淡，缺乏感情，这种为追求工业的高速发展而生产的建筑形态，与人们向往的自然生态环境格格不入。然而，壁画可作为人们面对自然的一个窗口，能在大厦的外墙和中厅等看到描绘自然的壁画，无疑是一种心旷神怡的心灵感觉。同时，在各种壁画形式中，最耐久、最易清洗、最耐侵蚀、色彩最鲜艳，表现手法最多样的，当属陶瓷壁画。  

&emsp;&emsp;故而在现今社会，壁画作为一个既有历史意义又有当代艺术价值的艺术形式，其创作及历史保护就显得尤为重要！

# 项目介绍

&emsp;&emsp;本项目受[DS变沙画-PaddleHub的迁移训练style Transfer图像迁移](https://aistudio.baidu.com/aistudio/projectdetail/2012947?channelType=0&channel=0)启发，基于paddlehub中的预训练风格迁移模型msgnet进行迁移学习。msgnet是基于'Multi-style Generative Network for Real-time Transfer'的风格迁移模型，它能够对输入图片变换21种不同风格。具体可参见[msgnet](https://www.paddlepaddle.org.cn/hubdetail?name=msgnet&en_category=ImageEditing).

## 代码步骤

&emsp;&emsp;本项目基于PaddlePaddle 2.0.2,PaddleHub 2.0.4，其中PaddleHub 2.0.4为AIStudio平台预装版本，故无需额外更改PaddleHub版本。（其他PaddleHub版本可能报错，各位小伙伴要是在其他环境体验要记得检查自己的PaddlePaddle和PaddleHub版本嗷！）

### Step1: 安装msgnet


```python
!hub install msgnet==1.0.0
```

### Step2: 相关数据准备
&emsp;&emsp;在完成PaddlePaddle与PaddleHub核对，以及msgnet安装后，即可开始使用msgnet模型对MiniCOCO等数据集进行Fine-tune。下面解压本项目挂在的作为style的壁画图片数据集，并下载minicoco作迁移训练数据。

【关于数据集说明】壁画数据集为本人百度爬虫搜集而来，并做了相应清洗，有325张壁画图片。


```python
import os
if not os.path.exists('data/minicoco'):
    %cd /home/aistudio/data/
    !unzip -oq /home/aistudio/data/data102159/mural.zip -d mural
    !wget -q https://paddlehub.bj.bcebos.com/dygraph/datasets/minicoco.tar.gz
    !tar -zxvf  /home/aistudio/data/minicoco.tar.gz
    %cd /home/aistudio/
```

&emsp;&emsp;获取本项目所需的相关数据后，需要对数据进行处理，并按Paddlepaddle2.0 的格式建立dataset。本项目将整个mural文件夹中的图片作Style，但依据[沙画项目](https://aistudio.baidu.com/aistudio/projectdetail/2012947?channelType=0&channel=0)，单张图作style，而非整个所有图片作Style时loss下降较快。若要整个文件夹训练而不是指定某张图片，把styleImg=None即可。


```python
import os
from typing import Callable
import cv2
import paddle
import paddlehub as hub
import numpy as np
import paddlehub.vision.transforms as T
import paddlehub.env as hubenv
from paddlehub.vision.utils import get_img_file
import os

class MyMiniCOCO(paddle.io.Dataset):
    """
    Dataset for Style transfer. The dataset contains 2001 images for training set and 200 images for testing set.
    They are derived form COCO2014. Meanwhile, it contains 21 different style pictures in file "21styles".
    Args:
       transform(callmethod) : The method of preprocess images.
       mode(str): The mode for preparing dataset.
    Returns:
        DataSet: An iterable object for data iterating
    """
    def __init__(self, transform: Callable, mode: str = 'train',
    data1path='/home/aistudio/data',styleFolder='mural',styleImg=None):
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.file = 'train'
        elif self.mode == 'test':
            self.file = 'test'
        if styleImg:
            self.style_file =os.path.join(data1path,  styleFolder,styleImg)
            self.style=[self.style_file]
        else:        
            self.style_file =os.path.join(data1path,  styleFolder)
            self.style = get_img_file(self.style_file)

        self.file = os.path.join(data1path, 'minicoco', self.file)
        self.data = get_img_file(self.file)
        assert (len(self.style)>0 and len(self.data)>0)
        print('self.data',len(self.data))
        print('self.style',len(self.style))


    def getImg(self,group,idx):
        im=[]
        ii=idx
        while len(im)==0:
            try:
                
                im = self.transform(group[ii])
            except :
                
                print('v',len(group),ii)
            ii-=1
        return im
    def __getitem__(self, idx: int) -> np.ndarray:

        im = self.getImg(self.data,idx)
        im = im.astype('float32')
        style_idx = idx % len(self.style)
        style = self.getImg(self.style,style_idx)
        style = style.astype('float32')
        return im, style

    def __len__(self):
        return len(self.data)

transform = T.Compose([T.Resize((256, 256), interpolation='LINEAR')])

styledata = MyMiniCOCO(transform,mode='train',data1path='/home/aistudio/data',styleFolder='mural',styleImg=None)

```

### Step3: fine-tune训练


【细节部分】：  

1、加载预训练模型：  

`model = hub.Module(name='msgnet',load_checkpoint=checkpoint)`  

其中：	  

&emsp;&emsp;* name: 选择预训练模型的名字。  

&emsp;&emsp;* load_checkpoint: 是否加载自己训练的模型，若为None，则加载提供的模型默认参数。

2、 选择优化策略和运行配置   
```
import paddle
from paddlehub.finetune.trainer import Trainer

optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
trainer = Trainer(model, optimizer,use_gpu=True,use_vdl=True, checkpoint_dir='test_style_ckpt')
trainer.train(styledata, epochs=200, batch_size=32, eval_dataset=None, log_interval=10, save_interval=1)
```
详情可见[官方PaddleHub finetune风格迁移教程](https://aistudio.baidu.com/aistudio/projectdetail/2231604)  

3、 额外说明

&emsp;&emsp;由于backbone部分其实不需要迁移，有可能影响迁移效果，故而下载paddle.vision中的vgg16，打印看看他有26组参数，所以设置msgnet的第25个之前的参数stop_gradient=True。


```python
import paddle
import paddlehub as hub
from paddle.vision.models import vgg16
from paddlehub.finetune.trainer import Trainer

import paddlehub.vision.transforms as T
# paddle.set_device("gpu:0")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %cd /home/aistudio/
# backbone model
backbone = vgg16(pretrained=True,num_classes=-1)
print('vgg parameters nums:',len(list(backbone.named_parameters())))
del backbone
##
trainFlag=True
goOnTrain=True

model = hub.Module(name='msgnet',load_checkpoint=None)
    
print(type(model),' parameters nums:',len(model.parameters()))
    ##
for index,param in enumerate(model.parameters()):
     #print(param.name)        
    if index>25:
            param.stop_gradient=False
    else:
        param.stop_gradient=True

optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
trainer = Trainer(model, optimizer,use_gpu=True,use_vdl=True, checkpoint_dir='test_style_ckpt')
trainer.train(styledata, epochs=200, batch_size=32, eval_dataset=None, log_interval=10, save_interval=1)
```

### Step4: 模型保存

通过查看msgnet的module.py源码，我们需要修改之前finetune训练的模型名称。
![](https://ai-studio-static-online.cdn.bcebos.com/acb54fb94906452c8899341f547732949f0b88a6d4f2414d8438be41524a3693)
以下是修改模型名称的脚本，这里讲训练的第200个epoch转换保存了，具体的可以修改：


```python
def epochModel():
    !mkdir /home/aistudio/model
    !rm -rf /home/aistudio/model/msgnet
    !mkdir /home/aistudio/model/msgnet
    ## 看需要取哪个epoch生成模型
    !cp -r test_style_ckpt/epoch_200/* /home/aistudio/model/msgnet/
    ##
    !mv /home/aistudio/model/msgnet/model.pdopt  /home/aistudio/model/msgnet/style_paddle.pdopt
    !mv /home/aistudio/model/msgnet/model.pdparams  /home/aistudio/model/msgnet/style_paddle.pdparams
epochModel()
```

### Step5: 模型预测

模型准备完毕后就可以使用自己训练的模型来体验啦！

origin里是原图地址，style是作style的图片地址，save_path是经过风格迁移后的图片保存地址


```python
import paddle
import paddlehub as hub
model = hub.Module(name='msgnet')
model = hub.Module(directory='model/msgnet')
result = model.predict(origin=["test1.jpg"], style="style_demo_img1.jpg", visualization=True, save_path ='style_tranfer')
result = model.predict(origin=["test1.jpg"], style="style_demo_img2.jpg", visualization=True, save_path ='style_tranfer')
```

# 参考：
１. [官方PaddleHub finetune风格迁移教程](https://aistudio.baidu.com/aistudio/projectdetail/2231604)　　　　

２. [DS变沙画-PaddleHub的迁移训练style Transfer图像迁移](https://aistudio.baidu.com/aistudio/projectdetail/2012947?channelType=0&channel=0)　　

３. [Paddlehub msgnet官网](https://www.paddlepaddle.org.cn/hubdetail?name=msgnet&en_category=ImageEditing)
