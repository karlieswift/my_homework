## vgg16迁移学习训练数据EnglishFnt
***题目：使用卷积神经元网络CNN，对多种字体的26个大写英文字母进行识别。***
<br>数据集介绍：</br>
<br>1- 数据集来源于Chars74K dataset，本项目选用数据集EnglishFnt中的一部分。Chars74K dataset网址链接 http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/</br>
<br>2- A-Z共26种英文字母，每种字母对应一个文件夹（Sample011对应字母A, Sample012对应字母B,…, Sample036对应字母Z）；</br>
<br>3- Sample011到Sample036每个文件夹下相同字母不同字体的图片约1000张，PNG格式</br>
<br>注意：EnglishFnt案例的通道channel是1 需要在transforms里将通道修改，调用代码torchvision.transforms.Grayscale(1)。同时需要将vgg6的第一层卷积层进行修改为in_channels=1，代码：self.vgg16.features[0] = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))</br>
<br>通过vgg16模型进行迁移学习，首先把vgg16的卷积层全部冻结，对前三层卷积进行参数激活进行更新，同时修改自己的全连接神经网络部分，对vgg16的全连接神经网络进行替换</br>
