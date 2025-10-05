import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.pointnet_regression_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg

# 加深全连接
# class get_model(nn.Module):
#     def __init__(self,num_class,normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
#         # 更深的全连接层，带有残差连接
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 512)  # 相同大小用于残差连接
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(256, 128)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.fc5 = nn.Linear(128, num_class)
#         self.drop = nn.Dropout(0.3)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)  
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)

#         x = self.drop(F.relu(self.bn1(self.fc1(x))))
#         residual = x  # 保存用于残差连接
#         x = self.drop(F.relu(self.bn2(self.fc2(x))))
#         x = x + residual  # 添加残差连接
#         x = self.drop(F.relu(self.bn3(self.fc3(x))))
#         x = self.drop(F.relu(self.bn4(self.fc4(x))))
#         x = self.fc5(x)

#         return x, l3_points

# msg4096（深）
# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel

#         # 第一层: 采样4096点，使用多尺度分组
#         self.sa1 = PointNetSetAbstractionMsg(
#             npoint=4096,
#             radius_list=[0.05, 0.1, 0.2],
#             nsample_list=[16, 32, 48],
#             in_channel=in_channel,
#             mlp_list=[[32, 48, 64], [64, 96, 128], [96, 128, 160]]
#         )

#         # 第二层: 采样2048点，使用多尺度分组
#         self.sa2 = PointNetSetAbstractionMsg(
#             npoint=2048,
#             radius_list=[0.1, 0.2, 0.4],
#             nsample_list=[32, 48, 64],
#             in_channel=(64 + 128 + 160) + 3,
#             mlp_list=[[64, 96, 128], [128, 196, 256], [196, 256, 320]]
#         )

#         # 第三层: 采样1024点，使用多尺度分组
#         self.sa3 = PointNetSetAbstractionMsg(
#             npoint=1024,
#             radius_list=[0.2, 0.4, 0.6],
#             nsample_list=[48, 64, 80],
#             in_channel=(128 + 256 + 320) + 3,
#             mlp_list=[[128, 192, 256], [256, 384, 512], [384, 512, 640]]
#         )

#         # 第四层: 采样256点，使用多尺度分组
#         self.sa4 = PointNetSetAbstractionMsg(
#             npoint=256,
#             radius_list=[0.4, 0.6, 0.8],
#             nsample_list=[64, 80, 96],
#             in_channel=(256 + 512 + 640) + 3,
#             mlp_list=[[256, 384, 512], [512, 768, 1024], [768, 1024, 1280]]
#         )

#         # 第五层: 全局聚合
#         self.sa5 = PointNetSetAbstraction(
#             npoint=None,
#             radius=None,
#             nsample=None,
#             in_channel=(512 + 1024 + 1280) + 3,
#             mlp=[1024, 1536, 2048],
#             group_all=True
#         )

#         # 全连接层，逐层降维
#         self.fc1 = nn.Linear(2048, 1536)
#         self.bn1 = nn.BatchNorm1d(1536)
#         self.drop1 = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(1536, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.drop2 = nn.Dropout(0.4)

#         self.fc3 = nn.Linear(1024, 768)
#         self.bn3 = nn.BatchNorm1d(768)
#         self.drop3 = nn.Dropout(0.4)

#         self.fc4 = nn.Linear(768, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.drop4 = nn.Dropout(0.3)

#         self.fc5 = nn.Linear(512, 256)
#         self.bn5 = nn.BatchNorm1d(256)
#         self.drop5 = nn.Dropout(0.3)

#         self.fc6 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)
#         B, _, _ = xyz.shape

#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None

#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

#         x = l5_points.view(B, 2048)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.drop3(F.relu(self.bn3(self.fc3(x))))
#         x = self.drop4(F.relu(self.bn4(self.fc4(x))))
#         x = self.drop5(F.relu(self.bn5(self.fc5(x))))
#         x = self.fc6(x)

#         return x, l5_points

# msg4096(浅)
# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel

#         # 第一层: 采样4096点，使用多尺度分组
#         self.sa1 = PointNetSetAbstractionMsg(
#             npoint=4096,
#             radius_list=[0.05, 0.1, 0.2],
#             nsample_list=[16, 32, 48],
#             in_channel=in_channel,
#             mlp_list=[[32, 48, 64], [64, 96, 128], [96, 128, 160]]
#         )

#         # 第二层: 采样2048点，使用多尺度分组
#         self.sa2 = PointNetSetAbstractionMsg(
#             npoint=2048,
#             radius_list=[0.1, 0.2, 0.4],
#             nsample_list=[32, 48, 64],
#             in_channel=(64 + 128 + 160) + 3,
#             mlp_list=[[64, 96, 128], [128, 196, 256], [196, 256, 320]]
#         )

#         # 第三层: 采样1024点，使用多尺度分组
#         self.sa3 = PointNetSetAbstractionMsg(
#             npoint=1024,
#             radius_list=[0.2, 0.4, 0.6],
#             nsample_list=[48, 64, 80],
#             in_channel=(128 + 256 + 320) + 3,
#             mlp_list=[[128, 192, 256], [256, 384, 512], [384, 512, 640]]
#         )

#         # 第四层: 采样256点，使用多尺度分组
#         self.sa4 = PointNetSetAbstractionMsg(
#             npoint=256,
#             radius_list=[0.4, 0.6, 0.8],
#             nsample_list=[64, 80, 96],
#             in_channel=(256 + 512 + 640) + 3,
#             mlp_list=[[256, 384, 512], [512, 768, 1024], [768, 1024, 1280]]
#         )

#         # 第五层: 全局聚合
#         self.sa5 = PointNetSetAbstraction(
#             npoint=None,
#             radius=None,
#             nsample=None,
#             in_channel=(512 + 1024 + 1280) + 3,
#             mlp=[1024, 1536, 2048],
#             group_all=True
#         )

#         # 全连接层，逐层降维
#         self.fc1 = nn.Linear(2048, 1024)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.drop1 = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(1024, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.drop2 = nn.Dropout(0.4)

#         self.fc3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.drop3 = nn.Dropout(0.3)

#         self.fc4 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)
#         B, _, _ = xyz.shape

#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None

#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

#         x = l5_points.view(B, 2048)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.drop3(F.relu(self.bn3(self.fc3(x))))
#         x = self.fc4(x)

#         return x, l5_points


#msg2048
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 7 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # 第一层: 采样2048点 (原来是4096)
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=2048,  # 修改为2048
            radius_list=[0.05, 0.1, 0.2],
            nsample_list=[16, 32, 48],  
            in_channel=in_channel, 
            mlp_list=[[32, 48, 64], [48, 64, 96], [64, 96, 128]]
        )
        
        # 第二层: 采样512点 (原来是1024，适当减少以保持层级比例)
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512, 
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[32, 48, 64],  
            in_channel=(64+96+128)+3, 
            mlp_list=[[64, 96, 128], [96, 128, 192], [128, 192, 256]]
        )
        
        # 第三层: 采样128点 (原来是256)
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.6],
            nsample_list=[48, 64, 80],
            in_channel=(128+192+256)+3,
            mlp_list=[[128, 192, 256], [192, 256, 384], [256, 384, 512]]
        )
        
        # 第四层: 全局特征，保持不变
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=(256+384+512)+3,
            mlp=[512, 1024, 2048],
            group_all=True
        )

        # 全连接层架构保持不变
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.3)
        
        self.fc5 = nn.Linear(256, num_class)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)  
        B, _, _ = xyz.shape
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        x = l4_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)

        return x, l4_points

#ssg8192
# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel
        
#         # 第一层: 采样8192点，保留原始点云信息
#         self.sa1 = PointNetSetAbstraction(
#             npoint=8192, 
#             radius=0.1, 
#             nsample=32, 
#             in_channel=in_channel, 
#             mlp=[64, 96, 128], 
#             group_all=False
#         )
        
#         # 第二层: 采样4096点
#         self.sa2 = PointNetSetAbstraction(
#             npoint=4096, 
#             radius=0.2, 
#             nsample=48, 
#             in_channel=128 + 3, 
#             mlp=[128, 196, 256], 
#             group_all=False
#         )
        
#         # 第三层: 采样2048点
#         self.sa3 = PointNetSetAbstraction(
#             npoint=2048, 
#             radius=0.3, 
#             nsample=64, 
#             in_channel=256 + 3, 
#             mlp=[256, 384, 512], 
#             group_all=False
#         )
        
#         # 第四层: 采样1024点
#         self.sa4 = PointNetSetAbstraction(
#             npoint=1024, 
#             radius=0.4, 
#             nsample=96, 
#             in_channel=512 + 3, 
#             mlp=[512, 768, 1024], 
#             group_all=False
#         )
        
#         # 第五层: 采样256点
#         self.sa5 = PointNetSetAbstraction(
#             npoint=256, 
#             radius=0.6, 
#             nsample=128, 
#             in_channel=1024 + 3, 
#             mlp=[1024, 1536, 2048], 
#             group_all=False
#         )
        
#         # 第六层: 全局聚合
#         self.sa6 = PointNetSetAbstraction(
#             npoint=None, 
#             radius=None, 
#             nsample=None, 
#             in_channel=2048 + 3, 
#             mlp=[2048, 2048, 2048], 
#             group_all=True
#         )

#         # 更深的全连接层，逐层降维
#         self.fc1 = nn.Linear(2048, 1536)
#         self.bn1 = nn.BatchNorm1d(1536)
#         self.drop1 = nn.Dropout(0.5)
        
#         self.fc2 = nn.Linear(1536, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.drop2 = nn.Dropout(0.4)
        
#         self.fc3 = nn.Linear(1024, 768)
#         self.bn3 = nn.BatchNorm1d(768)
#         self.drop3 = nn.Dropout(0.4)
        
#         self.fc4 = nn.Linear(768, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.drop4 = nn.Dropout(0.3)
        
#         self.fc5 = nn.Linear(512, 256)
#         self.bn5 = nn.BatchNorm1d(256)
#         self.drop5 = nn.Dropout(0.3)
        
#         self.fc6 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)  
#         B, _, _ = xyz.shape
        
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
            
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
#         l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)
        
#         x = l6_points.view(B, 2048)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.drop3(F.relu(self.bn3(self.fc3(x))))
#         x = self.drop4(F.relu(self.bn4(self.fc4(x))))
#         x = self.drop5(F.relu(self.bn5(self.fc5(x))))
#         x = self.fc6(x)

#         return x, l6_points


# ssg4096
# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel

#         # 第一层: 采样4096点
#         self.sa1 = PointNetSetAbstraction(
#             npoint=4096,
#             radius=0.1,
#             nsample=32,
#             in_channel=in_channel,
#             mlp=[64, 96, 128],
#             group_all=False
#         )

#         # 第二层: 采样2048点
#         self.sa2 = PointNetSetAbstraction(
#             npoint=2048,
#             radius=0.2,
#             nsample=48,
#             in_channel=128 + 3,
#             mlp=[128, 196, 256],
#             group_all=False
#         )

#         # 第三层: 采样1024点
#         self.sa3 = PointNetSetAbstraction(
#             npoint=1024,
#             radius=0.3,
#             nsample=64,
#             in_channel=256 + 3,
#             mlp=[256, 384, 512],
#             group_all=False
#         )

#         # 第四层: 采样256点
#         self.sa4 = PointNetSetAbstraction(
#             npoint=256,
#             radius=0.4,
#             nsample=96,
#             in_channel=512 + 3,
#             mlp=[512, 768, 1024],
#             group_all=False
#         )

#         # 第五层: 全局聚合
#         self.sa5 = PointNetSetAbstraction(
#             npoint=None,
#             radius=None,
#             nsample=None,
#             in_channel=1024 + 3,
#             mlp=[1024, 1536, 2048],
#             group_all=True
#         )

#         # 全连接层，逐层降维
#         self.fc1 = nn.Linear(2048, 1536)
#         self.bn1 = nn.BatchNorm1d(1536)
#         self.drop1 = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(1536, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.drop2 = nn.Dropout(0.4)

#         self.fc3 = nn.Linear(1024, 768)
#         self.bn3 = nn.BatchNorm1d(768)
#         self.drop3 = nn.Dropout(0.4)

#         self.fc4 = nn.Linear(768, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.drop4 = nn.Dropout(0.3)

#         self.fc5 = nn.Linear(512, 256)
#         self.bn5 = nn.BatchNorm1d(256)
#         self.drop5 = nn.Dropout(0.3)

#         self.fc6 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)
#         B, _, _ = xyz.shape

#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None

#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

#         x = l5_points.view(B, 2048)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.drop3(F.relu(self.bn3(self.fc3(x))))
#         x = self.drop4(F.relu(self.bn4(self.fc4(x))))
#         x = self.drop5(F.relu(self.bn5(self.fc5(x))))
#         x = self.fc6(x)

#         return x, l5_points

        
# ssg简化2048
# class get_model(nn.Module):
#     def __init__(self, num_class, normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 7 if normal_channel else 3
#         self.normal_channel = normal_channel
        
#         # 第一层: 采用2048点而不是8192点，减少内存占用
#         self.sa1 = PointNetSetAbstraction(
#             npoint=2048, 
#             radius=0.1, 
#             nsample=32, 
#             in_channel=in_channel, 
#             mlp=[64, 96, 128], 
#             group_all=False
#         )
        
#         # 第二层: 1024点
#         self.sa2 = PointNetSetAbstraction(
#             npoint=1024, 
#             radius=0.2, 
#             nsample=48, 
#             in_channel=128 + 3, 
#             mlp=[128, 160, 256], 
#             group_all=False
#         )
        
#         # 第三层: 512点
#         self.sa3 = PointNetSetAbstraction(
#             npoint=512, 
#             radius=0.4, 
#             nsample=64, 
#             in_channel=256 + 3, 
#             mlp=[256, 320, 512], 
#             group_all=False
#         )
        
#         # 第四层: 直接全局聚合，跳过采样中间层
#         self.sa4 = PointNetSetAbstraction(
#             npoint=None, 
#             radius=None, 
#             nsample=None, 
#             in_channel=512 + 3, 
#             mlp=[512, 768, 1024], 
#             group_all=True
#         )
        
#         # 优化全连接层架构：减少层数和维度
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
        
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
        
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.drop3 = nn.Dropout(0.3)
        
#         self.fc4 = nn.Linear(128, num_class)

#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)  
#         B, _, _ = xyz.shape
        
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
        
#         # 使用torch.cuda.amp.autocast()来启用混合精度训练
#         with torch.cuda.amp.autocast(enabled=True):
#             l1_xyz, l1_points = self.sa1(xyz, norm)
#             l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#             l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#             l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
            
#             x = l4_points.view(B, 1024)
#             x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#             x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#             x = self.drop3(F.relu(self.bn3(self.fc3(x))))
#             x = self.fc4(x)
        
#         return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.sqrt((target[:,0]-pred[:,0])**2 + (target[:,1]-pred[:,1])**2 + (target[:,2]-pred[:,2])**2)
        return loss.mean()