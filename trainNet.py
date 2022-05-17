from sklearn import datasets
import torch
import torch.utils.data as Data
import dataprocess as dp
import os
import torch.optim as optim
import pointnet.pointnet.model as net
import torch
import torch.nn.functional as F
from numpy import*
from torch.utils.tensorboard import SummaryWriter


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2



BatchSize=14#每一次训练喂BatchSize组数据
EpochSize=100#训练几轮

modelPath='/data/tyw/code/mufan/model/'
targetsPath='/data/tyw/code/mufan/targets/'
patient_no=['69','70','74','75','76','83','91','92','93','94','96','98','100','158','159','160','162','164','165']

test_patient=['76','84','91','103','158','232']
train_patient=['69','70','74','75','83','89','90',
'92','93','94','96','98','100','108','109','156','157',
'159','160','162','164','165','228','234'] 

trainPath='/data/tyw/code/mufan/data/'
arch_kind=["上颌","下颌"]

if __name__ == '__main__':
    # test_patient,train_patient=data_split(patient_no,ratio=0.1, shuffle=True)
    #特征点个数
    targets_num=54
    
    dataprocessor=dp.dataProcessor(trainPath,targetsPath,train_patient,model='all')

    x=dataprocessor.getTrainData()
    y=dataprocessor.getLabels()
    torch_dataset=dp.MyDataset(x,y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BatchSize,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )
    num_batch = len(torch_dataset) / BatchSize
    
    classifier = net.PointNetDenseCls(k=targets_num, feature_transform=True)
    optimizer = optim.Adam(classifier.parameters(),amsgrad=True,lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

    ta_train=[]
    tl_train=[]
    train_writer = SummaryWriter('./logs')
    for epoch in range(EpochSize):
        i=0
        for input,target in loader:
            i=i+1
            input=input.transpose(2, 1)
            input = input.type(torch.FloatTensor).cuda()
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            
            pred, trans, trans_feat = classifier(input)

            loss = loss_fn(pred, target)
            loss.sum().backward()
            optimizer.step()

            scheduler.step()
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.sum().item()))

            tl_train.append(loss.sum().item())
            print('epch: {} | num: {}  '.format(epoch,i))
        
        train_writer.add_scalar('train_loss', mean(tl_train), epoch)
    #保存网络权重
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (modelPath, epoch))
    print(test_patient)
    # print(x)
    # print(y)