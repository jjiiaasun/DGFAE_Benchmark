import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import time
import pandas as pd
import os
from torchsummary import summary
import copy
from data_rgb.datamgr_rgb import SimpleDataManager
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from util import set_random_seed
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import random

#hyper parameter
num_epochs = 5
num_classes = 7
batch_size = 32
image_size = 224


base_file = 'E:\\...\\train_CDEF.json'  ## change to the true path
val_file = 'E:\\...\\test_B.json'
base_datamgr = SimpleDataManager(image_size, batch_size)
dataloaders = base_datamgr.get_data_loader(base_file, aug=False)
set_random_seed(seed=0)

def pom1b_loss(output, target):

    batch_size, num_classes = output.size()

    # Convert target to one-hot encoding
    target_one_hot = torch.zeros(batch_size, num_classes).to(output.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    # Compute the loss
    loss = 0
    for k in range(num_classes):
        # Calculate P_i(k-l) for l in {-1, 0, 1}
        prob_sum = 0
        for l in [-1, 0, 1]:
            kl = k - l
            if 0 <= kl < num_classes:
                log_prob = torch.log(output[:, kl]+ 1e-10)
                prob_sum += log_prob

        # Update the loss
        loss += -torch.sum(target_one_hot[:, k] * prob_sum) # * class_weights[k]

    return loss / batch_size  # Normalize by batch size

def coral_loss(source, target):
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    source_cov = (source - source_mean).T @ (source - source_mean) / (source.size(0) - 1)
    target_cov = (target - target_mean).T @ (target - target_mean) / (target.size(0) - 1)
    return torch.mean((source_mean - target_mean) ** 2) + torch.mean((source_cov - target_cov) ** 2)


###Train
def train_model(model,dataloaders,criterion,optimizer,num_epochs,train_rate):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_valid=[]
    epoch_acc_valid=[]
    epoch_loss_train=[]
    epoch_acc_train=[]
    since=time.time()

    batch_num = len(dataloaders)
    train_batch_num = round(batch_num * train_rate)

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs - 1))
        print('-'*10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        accumulated_features = {label: None for label in range(7)}
        for step, (b_x, b_y) in enumerate(dataloaders):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:
                model.train()
                output, feature = model(b_x)
                output_probs = F.softmax(output, dim=1)
                loss_pom = pom1b_loss(output_probs, b_y)

                pre_lab = torch.argmax(output,1)
                loss_cri = criterion(output, b_y)

                coral_loss_value = 0
                for label in range(7):
                    src_feature_min = output[b_y == label]
                    if accumulated_features[label] is not None:
                        accumulated_feature = accumulated_features[label]
                        indices = torch.randperm(accumulated_feature.size(0))[:src_feature_min.size(0)]
                        sampled_accumulated_feature = accumulated_feature[indices]
                        coral_loss_num = coral_loss(src_feature_min, sampled_accumulated_feature)
                        if not torch.isnan(coral_loss_num).item():
                            coral_loss_value += coral_loss_num
                            # print(coral_loss_value)

                    if accumulated_features[label] is None:
                        accumulated_features[label] = src_feature_min.detach()
                    else:
                        accumulated_features[label] = torch.cat((accumulated_features[label], src_feature_min.detach()),
                                                                dim=0)

                loss = loss_pom  + 1 * loss_cri + 2 * coral_loss_value
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                with torch.no_grad():
                    output, _ = model(b_x)

                    output_probs = F.softmax(output, dim=1)  # 转换为概率分布
                    loss = pom1b_loss(output_probs, b_y)
                    pre_lab = torch.argmax(output,1)
                    val_loss += loss.item() * b_x.size(0)
                    val_corrects += torch.sum(pre_lab == b_y.data)
                    val_num += b_x.size(0)
        ### 计算一个epoch在训练集和验证集的损失和精度
        epoch_loss_train.append(train_loss / train_num)
        epoch_acc_train.append(train_corrects.double().item() / train_num)
        epoch_loss_valid.append(val_loss / val_num)
        epoch_acc_valid.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(
            epoch, epoch_loss_train[-1], epoch_acc_train[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
            epoch, epoch_loss_valid[-1], epoch_acc_valid[-1]))
        ### 拷贝模型最高精度下的参数
        if epoch_acc_valid[-1] > best_acc:
            best_acc = epoch_acc_valid[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        ### 使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "epoch_loss_train": epoch_loss_train,
              "epoch_loss_val": epoch_loss_valid,
              "epoch_acc_train": epoch_acc_train,
              "epoch_acc_val": epoch_acc_valid})
    return model, train_process

class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = models.resnet18(pretrained=True)

        # 修改最后的全连接层，用于输出指定类别数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # 提取ResNet18的特征提取部分（去除最后的fc层）
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        # 提取特征向量（不通过最后的fc层）
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # 将特征展平成(batch_size, 512)

        # 获取最终的预测结果
        pred = self.resnet.fc(features)

        # 返回预测结果和特征向量
        return pred, features

####### 迁移学习 Resnet18
net = ResNet18()
learning_rate = 0.0001
optimizer = AdamW(net.parameters(), lr=learning_rate) #, weight_decay=weight_decay
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=num_epochs * len(dataloaders),
                       pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                       base_momentum=0.85, max_momentum=0.95)


# # 判断计算机的GPUs是否可用
device = ('cuda' if torch.cuda.is_available()
          else 'cpu')
print(torch.cuda.is_available())
net = net.to(device)
criterion = nn.CrossEntropyLoss()
net, train_process = train_model(net, dataloaders, criterion, optimizer, num_epochs, 0.8)
### 存储模型参数
torch.save(net.state_dict(),'save_ResNet_params.pth')
### 存储训练结果
with pd.ExcelWriter('train_process.xlsx') as writer:
    train_process.to_excel(writer, sheet_name='train_process')


dataloaders = base_datamgr.get_data_loader(val_file, aug=False)
epoch_loss_test = []
epoch_acc_test = []
epoch_acc_tr_test = []
test_loss = 0.0
test_corrects = 0
test_corrects_true = 0
test_num = 0
label = []
pre = []

for step, (bt_x, bt_y) in enumerate(dataloaders):
    bt_x = bt_x.to(device)
    bt_y = bt_y.to(device)
    net.eval()
    outputs = []

    with torch.no_grad():
        output, _ = net(bt_x)
        pre_lab = torch.argmax(output, 1)
        loss = criterion(output, bt_y)
        test_loss += loss.item() * bt_x.size(0)
        test_corrects_true += torch.sum(pre_lab == bt_y.data)
        test_corrects += torch.sum(pre_lab == bt_y.data)
        test_corrects += torch.sum(pre_lab == bt_y.data - 1)
        test_corrects += torch.sum(pre_lab == bt_y.data + 1)
        test_num += bt_x.size(0)
        epoch_loss_test.append(test_loss / test_num)
        epoch_acc_test.append(test_corrects.double().item() / test_num)
        epoch_acc_tr_test.append(test_corrects_true.double().item() / test_num)
        test_process = pd.DataFrame(
            data={"epoch_loss_test": epoch_loss_test,
                  "epoch_acc_test": epoch_acc_test,
                  "epoch_acc_tr_test": epoch_acc_tr_test})
        bt_y = bt_y.tolist()
        pre_lab = pre_lab.tolist()
        label.append(bt_y)
        pre.append(pre_lab)
label_lst = sum(label, [])
pre_lst = sum(pre, [])
label_np = np.array(label_lst)
pre_np = np.array(pre_lst)

### 存储测试结果
with pd.ExcelWriter('test_process.xlsx') as writer:
    test_process.to_excel(writer, sheet_name='test_process')

### 绘制混淆矩阵
conf_mat = confusion_matrix(label_np, pre_np)
sns.heatmap(conf_mat, annot=True, fmt='.20g', cmap="Blues")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_map.svg', format='svg', dpi=500)  # 修改为SVG格式
plt.show()

