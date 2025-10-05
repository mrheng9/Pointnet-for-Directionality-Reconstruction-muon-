import torch
import torch.nn as nn
import math  
import torch.nn.functional as F
import pandas as pd
from models.pointnet_regression_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 9 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # 加载PMT类型数据
        df = pd.read_csv('/disk_pool1/houyh/data/PMTType_CD_LPMT.csv', sep=' ')
        df_hama = df[df['type'] == 'Hamamatsu']
        df_nnvt = df[df['type'].isin(['HighQENNVT', 'NNVT'])]
        
        # 获取索引
        list_0 = torch.tensor(df_hama['index'].tolist(), dtype=torch.long)
        list_1 = torch.tensor(df_nnvt['index'].tolist(), dtype=torch.long)
        
        # 注册缓冲区
        self.register_buffer('list_0', list_0)  # Hamamatsu PMT索引
        self.register_buffer('list_1', list_1)  # NNVT PMT索引
        
        # MSG配置的PointNet++架构
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.1, 0.2, 0.3], nsample_list=[32, 48, 64], 
                                            in_channel=in_channel, mlp_list=[[32, 64, 128], [64, 96, 192], [64, 128, 256]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.2, 0.4, 0.5], nsample_list=[64, 96, 128], 
                                            in_channel=(128+192+256)+3, mlp_list=[[128, 128, 256], [128, 192, 256], [128, 256, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=(256+256+256)+3, 
                                         mlp=[512, 1024, 2048], group_all=True)
        
        # MLP头部用于回归
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, num_class)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)  
        B, _, _ = xyz.shape
        
        # 根据PMT类型分别处理（这里仅添加分类逻辑，不实际处理PMT数据）
        if self.normal_channel:
            # 将输入特征分为位置和其他特征
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
            
            # 这里可以根据PMT类型添加不同的处理逻辑
            # 例如可以根据索引区分哈马马츠和NNVT类型的PMT
            # 注意：这里只添加了分类逻辑，没有添加CNNBlock处理
        else:
            norm = None
        
        # PointNet++特征提取
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # MLP回归头部
        x = l3_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        # 欧式距离损失
        loss = torch.sqrt((target[:,0]-pred[:,0])**2 + (target[:,1]-pred[:,1])**2+ (target[:,2]-pred[:,2])**2)
        return loss.mean()