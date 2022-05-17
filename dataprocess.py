from tabnanny import check
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

import vedo
import os
import re
#读取stl文件并进行数据预处理
from numpy import dtype, random


class MyDataset(Dataset):
    def __init__(self,x,y):
        self.data = x
        self.label = y

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


class dataProcessor():
    def __init__(self,trainPath,targetPath,patients_no,sample_nums=500,model='all'):
        self.pointsPath=targetPath
        self.teethPath=trainPath
        self.patients_no=patients_no
        self.sample_nums=sample_nums
        if model=='front':
            self.targets_nums=14
            self.teeth_num=6
        elif model=='all':
            self.targets_nums=54
            self.teeth_num=14
        self.available_patients_no=[]
        self.upperArchesPoints=[] #记录所有上牙列的特征点N×54×3
        self.checkTargets() #获取相应的特征点记录在upperArchesPoints
        


        self.upperArchesMesh15=[]
        self.x = torch.zeros(len(self.available_patients_no)*self.teeth_num,self.sample_nums,15)
        self.y = torch.zeros(len(self.available_patients_no)*self.teeth_num,self.sample_nums,self.targets_nums)
        self.centerMatrix=np.zeros([len(self.available_patients_no)*self.teeth_num,self.sample_nums,3],dtype=float)
        
        #创建输入矩阵M个病人14个牙齿1k网格 获取x
        for i in range(len(self.available_patients_no)):
            patient_no=self.patients_no[i]
            self.read_one_patient_arch(patient_no,i)

        
        
    def getCenter(self,count,teeth_index,row):
        return self.centerMatrix[count*self.teeth_num+teeth_index][row]

    def checkTargets(self):
        patients_no=self.patients_no
        for i in range(len(self.patients_no)):
            checkUpper,upperTargets=self.checkTarget(patients_no[i])
            if checkUpper!=True:
                print('fail'+patients_no[i])
                
            else:
                self.available_patients_no.append(patients_no[i])
                self.upperArchesPoints.append(upperTargets)
        return 

    #指明一个病人
    #读取出特征点文件igs 返回特征点是否符合规范 是返回true 并设置self.upper_targets
    #[upper_targets,lower_targets]
    def checkTarget(self,patient_no):

        path_target=self.pointsPath
        upperCheck=True
        lowerCheck=True
        #获得一个该目录下各个/patient_no/filename的列表[[上牙弓的文件名],[下牙弓的文件名]]
        res=self.readTargetFilePath(patient_no)
        upper=res[0]#记录上牙列的igs文件
        lower=res[1]#记录下牙列的igs文件
        if len(upper)!=self.teeth_num:
            upperCheck=False
        if len(lower)!=self.teeth_num:
            lowerCheck=False

        upper_targets=np.zeros([self.targets_nums,3],dtype=float) #记录上牙列54个特征点
        count=0#记录当前牙齿个数
        #遍历所有的上牙弓
        # print(patient_no)
        # print('---------')
        for teeth in upper: 
            if upperCheck==False:
                break
            #读取出igs文件的特征点
            points_teeth=self.readData(path_target+'/'+patient_no,teeth)
            for point in points_teeth:
                upper_targets[count][0:]=point
                # upper_targets.append(point.copy())
                count+=1
            # print(teeth)
            # print(len(points_teeth)) 
        if count!=self.targets_nums or upperCheck==False:
            print("upper fail"+str(patient_no))
            upperCheck=False
        else:
            upper_targets=upper_targets
            upperCheck=True
        
        return upperCheck,upper_targets

    def getTargetPoints(self,patient_no):
        return self.upperArchesPoints[patient_no]


    #读取一个特征点目录文件下的igs文件名称
    def readTargetFilePath(self,patient_no):
        #把牙齿个数不符合一列14个的牙弓剔除
        cur_path=self.pointsPath+'/'+patient_no+'/'
        upper=[]
        lower=[]
        for file in os.listdir(cur_path):#牙齿文件
            #只取前牙部分
            if self.teeth_num==6:
                if file.endswith("1.igs") or file.endswith("2.igs") or file.endswith("3.igs"):
                    if file.startswith("1") or file.startswith("2") :
                        upper.append(file)
                    elif file.startswith("3") or file.startswith("4"):
                        lower.append(file)
            else:
                if file.endswith("igs") :
                    if file.startswith("1") or file.startswith("2") :
                        upper.append(file)
                    elif file.startswith("3") or file.startswith("4"):
                        lower.append(file)
        upper.sort()
        return [upper,lower]

    #参数指明特征点文件目录和病人序号
    #返回一个牙齿的特征点
    def readData(self,path,teeth_name):
        f =open(path+'/'+teeth_name,mode='r')
        pattern="^116,"#表明该行记录的是点坐标
        points_num=0
        points=[]
        content=f.readlines() 
        #提取出特征点位置信息到Points 倒着读使得点序正确
        for con in reversed(content):
        #     print(con)
            #找到记录特征点位置的行
            if re.match(pattern,con):
                points_num+=1
                #选取出点的位置
                point=con.split(";")[0].split(",")[1:4]
                #单个特征点坐标进行处理 转为小数
                coord=np.zeros(3,dtype=float)
                for i in range(len(point)):
                    p=point[i]
                    split_p=p.split('D')
                    data=float(split_p[0])
                    index=int(split_p[1])
                    #D的指数部分处理
                    if index>=0:
                        negative=False
                    else:
                        negative=True
                        index=-index
                    for j in range(index):
                        if negative:
                            # print('negative')
                            data/=10
                        else:
                            data*=10
                    coord[i]=data
                points.append(coord)#加入一个点的坐标
    #     print(points)
        f.close()
        return points

    #读一个病人14个上牙弓牙齿的N×15矩阵
    #patient_no病人序列号 病人序号在列表中的序号
    def read_one_patient_arch(self,patient_no,count):
        stlNames=[]

        #读取filename路径下的stl数据集合
        filename=self.teethPath+"/"+patient_no+"/"+'上颌/'
        for file in os.listdir(filename):
            #仅输入前牙文件
            if self.teeth_num==6:
                if file.endswith("1.stl") or file.endswith('2.stl') or file.endswith('3.stl'):
                    stlNames.append(file)
            else:
                if file.endswith("stl"):
                    stlNames.append(file)
        stlNames.sort()
        
        # stls_result=np.zeros([len(stlNames),15,self.sample_nums], dtype='float32')
        # stls_result = torch.from_numpy(stls_result)
        for i in range(len(stlNames)):
            stl=stlNames[i]
            self.stlName=stl
            mesh=vedo.load(filename + stl)
            # print(mesh.NCells())
            # #对单个牙齿进行数据处理
            res,cells=self.preprocess(mesh)
            while(res.shape[1]>self.sample_nums
            ):
                j=random.randint(0,res.shape[1]-1)
                res=np.delete(res, j, 1)#删除对应的列
                cells=np.delete(cells,j,0)#删除对应的三角形中心点
            # numpy -> torch.tensor
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            X = res.transpose(1, 0)
            X = torch.from_numpy(X).to('cuda', dtype=torch.float)
            self.centerMatrix[count*self.teeth_num+i]=cells
            self.x[count*self.teeth_num+i]=X
            #计算y 参数mesh和特征点targets的坐标
            row=self.gaussHeatMap(cells,self.upperArchesPoints[count])
            self.y[count*self.teeth_num+i]=torch.from_numpy(row)
            # stls_result[i]=X
        return 
        # return stls_result

    #对单个牙齿的网格进行15维度运算
    def preprocess(self,mesh):
        # mesh = vedo.load("11.stl")
        mesh_d = mesh.clone()
        sample_nums=self.sample_nums
        # pre-processing: downsampling
        if mesh.NCells() > sample_nums:
            # print('\tDownsampling...')
            target_num = sample_nums
            for i in range(10):
                ratio = (target_num + i) / mesh.NCells()
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
                if mesh_d.NCells() >= sample_nums:
                    break

        else:
            print("false")
            mesh_d = mesh.clone()

        # print(f'Simplified mesh has {mesh_d.NPoints()} vertices and {mesh_d.NCells()} triangles')

        # move mesh to origin
        # print('\tPredicting...')
        cells = np.zeros([mesh_d.NCells(), 9], dtype='float32')
        polydata = mesh_d.polydata()
        for i in range(len(cells)):
            cells[i][0], cells[i][1], cells[i][2] = polydata.GetPoint(
                polydata.GetCell(i).GetPointId(0))  # don't need to copy
            cells[i][3], cells[i][4], cells[i][5] = polydata.GetPoint(
                polydata.GetCell(i).GetPointId(1))  # don't need to copy
            cells[i][6], cells[i][7], cells[i][8] = polydata.GetPoint(
                polydata.GetCell(i).GetPointId(2))  # don't need to copy
        #原坐标的位置
        original_cells_d = cells.copy()
        #中心点的坐标mean_cell_centers
        mean_cell_centers = mesh_d.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
        v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 3] - cells[:, 6]
        v2[:, 1] = cells[:, 4] - cells[:, 7]
        v2[:, 2] = cells[:, 5] - cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        mesh_d.addCellArray(mesh_normals, 'Normal')

        # preprae input
        points = mesh_d.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = mesh_d.getCellArray('Normal').copy()  # need to copy, they use the same memory address
        barycenters = mesh_d.cellCenters()  # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        # X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))

        # print(X)
        return X.T,mesh_d.cellCenters()

    #计算一个牙齿的目标高斯热图
    #参数：网格N×9  targets:54×3
    # 计算Y N×54 
    def gaussHeatMap(self,cells_center,targets):
        H=1.0
        sigma=5
        res=np.zeros([len(cells_center),len(targets)],dtype=float)
        col=0
        for xL in targets:#t 一个坐标
            row=0
            for x in cells_center:#取三角形的中心点 一个坐标
                z=0.0
                cur=0.0
                dis=x-xL
                for i in range(3):
                    cur+=dis[i]**2
                # print(cur)
                cur=-cur/2
                z=H*np.exp(cur/(sigma**2))
                if z < 1e-6:
                    z=0
                res[row][col]=z
                row+=1

            col+=1
        return res

    def getLabels(self):
        return self.y

    def getTrainData(self):
        return self.x






if __name__ == '__main__':
    root="/data/tyw/code/mufan/sample_data/"
    patient_no="1"
    arch_kind=["上颌","下颌"]

    #读取filename路径下的stl数据集合
    filename=root+"/"+patient_no+"/"+arch_kind[0]
    print(filename)
    stlnames=[]
    for file in os.listdir(filename):
        if file.endswith(".stl"):
            stlnames.append(file)
    mesh = vedo.load("/data/tyw/tooth data/20220321牙列数据/1/上颌/11.stl")
    print("mesh 网格数")
    print(mesh.NCells())
    mesh_d = mesh.clone()
    if mesh.NCells() != 3000:
        print("网格数错误")


    # pre-processing: downsampling
    if mesh.NCells() > 10000:
        print('\tDownsampling...')
        target_num = 10000
        ratio = target_num / mesh.NCells()  # calculate ratio
        mesh_d = mesh.clone()
        mesh_d.decimate(fraction=ratio)
        predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
    else:
        mesh_d = mesh.clone()
        predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
    # move mesh to origin
    print('\tPredicting...')
    cells = np.zeros([mesh_d.NCells(), 9], dtype='float32')
    polydata = mesh_d.polydata()
    for i in range(len(cells)):
        cells[i][0], cells[i][1], cells[i][2] = polydata.GetPoint(
            polydata.GetCell(i).GetPointId(0))  # don't need to copy
        cells[i][3], cells[i][4], cells[i][5] = polydata.GetPoint(
            polydata.GetCell(i).GetPointId(1))  # don't need to copy
        cells[i][6], cells[i][7], cells[i][8] = polydata.GetPoint(
            polydata.GetCell(i).GetPointId(2))  # don't need to copy

    original_cells_d = cells.copy()

    mean_cell_centers = mesh_d.centerOfMass()
    cells[:, 0:3] -= mean_cell_centers[0:3]
    cells[:, 3:6] -= mean_cell_centers[0:3]
    cells[:, 6:9] -= mean_cell_centers[0:3]

    # customized normal calculation; the vtk/vedo build-in function will change number of points
    v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
    v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
    v1[:, 0] = cells[:, 0] - cells[:, 3]
    v1[:, 1] = cells[:, 1] - cells[:, 4]
    v1[:, 2] = cells[:, 2] - cells[:, 5]
    v2[:, 0] = cells[:, 3] - cells[:, 6]
    v2[:, 1] = cells[:, 4] - cells[:, 7]
    v2[:, 2] = cells[:, 5] - cells[:, 8]
    mesh_normals = np.cross(v1, v2)
    mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
    mesh_normals[:, 0] /= mesh_normal_length[:]
    mesh_normals[:, 1] /= mesh_normal_length[:]
    mesh_normals[:, 2] /= mesh_normal_length[:]
    mesh_d.addCellArray(mesh_normals, 'Normal')

    # preprae input
    points = mesh_d.points().copy()
    points[:, 0:3] -= mean_cell_centers[0:3]
    normals = mesh_d.getCellArray('Normal').copy()  # need to copy, they use the same memory address
    barycenters = mesh_d.cellCenters()  # don't need to copy
    barycenters -= mean_cell_centers[0:3]

    # normalized data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))
    # X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))
    print(X)
    print(X.shape)

