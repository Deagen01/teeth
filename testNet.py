import numpy as np
from sklearn import datasets
import torch
import torch.utils.data as Data
import dataprocess as dp
import os
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import torch.optim as optim
import pointnet.pointnet.model as net
import torch
import torch.nn.functional as F
from numpy import*
import math
from torch.utils.tensorboard import SummaryWriter

outputPath='/data/tyw/code/mufan/pred_res/'


modelName='seg_model_99v1.pth'
modelPath='/data/tyw/code/mufan/model/'
targetsPath='/data/tyw/code/mufan/targets/'
patient_no=['69','70','74','75','76','83','91','92','93','94','96','97','98',
'100','157','158','159','160','161','162','164','165']
outputPath='data/tyw/code/mufan/res/'

trainPath='/data/tyw/code/mufan/data/'
arch_kind=["上颌","下颌"]
test_patient=['76','84','91','103','158','232'] #76
sample_nums=500
distThreshold=6
BatchSize=14#每一次训练喂BatchSize组数据

#记录特征点对应的列
target_num=[2,2,3,6,6,4,4]
target_col=[[0,2],[2,4],[4,7],[7,13],[13,19],[19,23],[23,27],
            [27,29],[29,31],[31,34],[34,40],[40,46],[46,50],[50,54]]


def writeTarget(index,point,f):
    x=str(point[0])
    y=str(point[1])
    z=str(point[2])
    f.write(str(index)+': '+x+','+y+','+z+' ')
#计算两个坐标的欧氏距离的平方    
def calDist(a,b):
    res=0.0
    for i in range(3):
        res+=(a[i]-b[i])**2
    return res

if __name__ == '__main__':
    
    targets_num=54

    f = open('./pred_res/res.txt','w')
    f_dist = open('./pred_res/dist.txt','w')
    heatMap=np.zeros([len(test_patient),sample_nums])
    dataprocessor=dp.dataProcessor(trainPath,targetsPath,test_patient,model='all')

    x=dataprocessor.getTrainData()
    y=dataprocessor.getLabels()


    torch_dataset=dp.MyDataset(x,y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BatchSize,
        shuffle=False
    )
    loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    # loss_fn=torch.nn.MSELoss(reduce=False, size_average=False)
    classifier = net.PointNetDenseCls(k=targets_num, feature_transform=True)
    classifier.load_state_dict(torch.load(modelPath+modelName),False)
    classifier.cuda()
    count=0
    test_writer = SummaryWriter('./logs')
    front_loss=0
    rear_loss=0
    loss_of_teeth=np.zeros(54,dtype=float)
    
    for input,target in loader:
        f.write('PatientNo'+test_patient[count]+'\n')
        f_dist.write('PatientNo'+test_patient[count]+'\n')
        input=input.transpose(2, 1)
        input = input.type(torch.FloatTensor).cuda()
        input, target = input.cuda(), target.cuda()
        
        classifier = classifier.eval()
        
        pred, trans, trans_feat = classifier(input)

        loss = loss_fn(pred, target)
        
        # Comparison between image_max and im to find the coordinates of local maxima
        
        res=pred.detach().cpu().numpy()
        teeth_index=0


        for res_i in res:#遍历batchsize个牙齿
            r=res_i.copy()
            print(r.max())
            
            f.write('teeth'+str(teeth_index)+'\n')
            col_left=target_col[teeth_index][0]
            col_right=target_col[teeth_index][1]
            # part_r=r[0:,col_left:col_right]
            part_r=r
            pred_centers=[]
            #得到牙列特征点的真实值
            arch_gt_points=dataprocessor.getTargetPoints(count)
            gt_points=[]
            marked_feature=[]
            for point in arch_gt_points[col_left:col_right]:
                gt_points.append(list(point))
            flag=True
            while(len(pred_centers)!=(col_right-col_left)):
            # for time in range(col_right-col_left):#遍历特征点个数
                #求出矩阵r的最大位置
                m = argmax(part_r)
                output_targets=[]
                row,col = divmod(m, part_r.shape[1])#计算出对应行的网格
                center=dataprocessor.getCenter(count,teeth_index,row)
                for last_center in pred_centers:
                    if calDist(last_center,center) < 10:
                        part_r[row][0:]=0#将该网格所在行置为0
                        flag=False
                        break#重新查找
                    else:
                        flag=True
                if flag==False:
                    continue
                pred_centers.append(center)

                #保存每一个牙齿的特征点真实值与目标值的距离平方和
                dist_pred_gt=[]
                
                for index in range(len(gt_points)):
                    # np_center=np.array(center)
                    # np_gt_point=np.array(gt_points[index])
                    # print(np_center-np_gt_point)
                    if index in marked_feature:
                        dist_pred_gt.append(100)
                        continue
                    dist_pred_gt.append(sqrt(calDist(center,gt_points[index])))
                
                feature_index=dist_pred_gt.index(min(dist_pred_gt))
                marked_feature.append(feature_index)

                gt_point=gt_points[feature_index]#对应的特征点真是值
                
                print(str(teeth_index)+'dist')
                
                loss_point=dist_pred_gt[feature_index]
                loss_of_teeth[col_left+feature_index]+=loss_point
                # f_dist.write('teeth_index'+str(feature_index)+'\n'+str(loss_point)+'\n')
                print(loss_point)
                print('------------')
                

                # gt_points=np.delete(gt_points,feature_index)
                #写入预测特征点位置
                writeTarget(row,center,f)
                #将对应列的target置为0
                # part_r[0:,col]=0
                part_r[row,0:]=0

            f.write('\n')
            teeth_index+=1




        print('test loss: %f' % (loss.sum().item()))
        test_writer.add_scalar('test_loss', loss.sum().item(), count)
        count+=1
    indexOfL=1
    for l in loss_of_teeth:
        l=l/len(test_patient)
        f_dist.write('teeth'+str(indexOfL)+':')
        f_dist.write(str(l)+'\n')
        indexOfL+=1
    f.close()
    f_dist.close()
    
    
        
        