import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import torch.nn as nn
from torch.optim import lr_scheduler
import time

'''
模型说明：
输入：大文件夹名字；大文件夹--小文件夹（小文件夹名字为类别名字）
训练次数：epochs
'''
# 填写
Data = "DARK"
epochs = int(30)
model_name = int(2024090501)
print(model_name)
Data = Data
# 调取文件夹，每个分类建立小文件夹，小文件夹名称是类别名

a = r'./%s/*/*.jpg' % Data
imgs_path = glob.glob(a)
# 提取一张图片的类别名称
img_p = imgs_path[1]
img_p.split('\\')[1].split('.')[0]  # 按照（）标志后面的第几个即可
labels_name = [img_p.split('\\')[1].split('.')[0] for img_p in imgs_path]
unique_labels = np.unique(labels_name)
# 每一个类别映射到数字
label_to_index = dict((v, k) for k, v in enumerate(unique_labels))
index_to_label = dict((v, k) for k, v in label_to_index.items())
all_labels = [label_to_index.get(name) for name in labels_name]
print(len(imgs_path), len(all_labels), len(index_to_label), len(label_to_index))

# 图片乱序排列, 先得到乱序的index，再按照这个index进行排序
np.random.seed(2021)
random_index = np.random.permutation(len(imgs_path))    # 做乱序的index
imgs_path = np.array(imgs_path)[random_index]           # 图片按照index排序
all_labels = np.array(all_labels)[random_index]         # 乱序图片的labels
print(imgs_path[5:])

# 创建dataloader
# 划分数据集 tain和test
i = int(len(imgs_path)*0.8)
train_path = imgs_path[: i]
train_labels = all_labels[: i]
test_path = imgs_path[i :]
test_labels = all_labels[i :]
transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor()
                              ])
test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[.5, .5, .5],
                                                     std=[.5, .5, .5])
                               ])
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(256, scale=(0.6,1.0), ratio=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                              ])

# 建立训练模型
class Train_Mydataset(data.Dataset):
    def __init__(self, imgs_path, labels):  # 图片路径 及其标签
        self.imgs = imgs_path
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index]  # 切片
        label = self.labels[index]  # 切片的label

        pil_img = Image.open(img).convert('RGB')  # 读取图片

        # 改之前
        # img_data = np.asarray(pil_img, dtype=np.uint8)
        # img_data = transform(Image.fromarray(img_data))

        img_data = transform(pil_img)  # 改之后

        return img_data, label

    def __len__(self):
        return len(self.imgs)

# 建立测试模型
class Test_Mydataset(data.Dataset):
    def __init__(self, imgs_path, labels):  # 图片路径 及其标签
        self.imgs = imgs_path
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index]  # 切片
        label = self.labels[index]  # 切片的label

        pil_img = Image.open(img).convert('RGB')  # 读取图片

        # 改之前
        # img_data = np.asarray(pil_img, dtype=np.uint8)
        # img_data = transform(Image.fromarray(img_data))

        img_data = transform(pil_img)  # 改之后

        return img_data, label

    def __len__(self):
        return len(self.imgs)

# 训练集和测试集
train_ds = Train_Mydataset(train_path, train_labels)
test_ds = Test_Mydataset(test_path, test_labels)

BATCH_SIZE = 16    # 数据已是乱序，无需再打乱
train_dl = data.DataLoader(
                      train_ds,
                      batch_size=BATCH_SIZE,)
test_dl = data.DataLoader(
                      test_ds,
                      batch_size=BATCH_SIZE,)

# 建立图片与其标签
img_batch, label_batch = next(iter(train_dl))

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, 4)  # 三分类

if torch.cuda.is_available():
    model.to('cuda')
# 建立损失函数
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)  #原来是0.0001 0.01
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')

        y = torch.tensor(y, dtype=torch.long)  # 抓取的dict的标签

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')

            y = torch.tensor(y, dtype=torch.long)  # 抓取的dict的标签

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = epochs

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

plt.plot(range(1, len(train_loss)+1), train_loss, label='train_loss')
plt.plot(range(1, len(train_loss)+1), test_loss, label='test_loss')
plt.legend()

plt.plot(range(1, len(train_loss)+1), train_acc, label='train_acc')
plt.plot(range(1, len(train_loss)+1), test_acc, label='test_acc')
plt.legend()

# 保存预训练权重在当前目录下
PATH = '%s.pth' %model_name
torch.save(model.state_dict(), PATH)   # 保存最高正确率的权重



