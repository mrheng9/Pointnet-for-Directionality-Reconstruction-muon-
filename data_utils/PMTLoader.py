import os
import numpy as np
import warnings
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PMTDataLoader(Dataset):
    def __init__(self, points, labels, args):
        """
        简化版PMT数据加载器 - 无预处理步骤
        
        参数:
            points (np.ndarray): 点云数据，形状为(N, num_points, feature_dims)
            labels (np.ndarray): 目标标签
            args: 包含配置参数的参数对象
        """
        self.points = points
        self.labels = labels
        self.use_normals = args.use_normals if hasattr(args, 'use_normals') else False
        self.normalize_points = args.normalize_points if hasattr(args, 'normalize_points') else False
        
        # 记录原始点云形状
        self.original_npoints = self.points.shape[1] if len(self.points.shape) > 2 else 0
        print(f"点云形状: {self.points.shape}")
        
    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # 获取当前样本
        point_set = self.points[index].copy()  # 使用copy避免修改原始数据
        label = self.labels[index]
        
        # 标准化点云坐标 (可选)
        if self.normalize_points:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        # 如果不使用法向量，只保留XYZ坐标 (可选)
        if not self.use_normals and point_set.shape[1] > 3:
            point_set = point_set[:, 0:3]
        
        # 转换为PyTorch张量
        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array(label).astype(np.float32))
        
        return point_set, label

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
def get_stacked_datach(_folder_path, _feature_name):
    _file_num = 1000
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/%s/x_%s_%i.npy' % (_folder_path, _feature_name, _feature_name, _i)):
           _x.append(np.load('%s/%s/x_%s_%i.npy' % (_folder_path, _feature_name, _feature_name,_i)))
    _x = np.concatenate(_x)
    # if _feature_name == 'pmt_fht':
    #     _x[_x == 1250] = 0
    if _feature_name == 'pmt_fht2':
        _x[_x >=900] = 0
        _x[_x < 0 ] = 0
    print(_feature_name + ' has been loaded ')
    return _x

def get_stacked_datachy(_folder_path, _feature_name):
    _file_num = 1000
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/%s_%i.npy' % (_folder_path, _feature_name, _i)):
            _x.append(np.load('%s/%s_%i.npy' % (_folder_path, _feature_name, _i)))
    _x = np.concatenate(_x)
    print(_feature_name + ' has been loaded')
    return _x

def get_stacked_datawei(_folder_path, _feature_name):
    _file_num = 1000
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/x_%s_%i.npy' % (_folder_path,  _feature_name, _i)):
           _x.append(np.load('%s/x_%s_%i.npy' % (_folder_path, _feature_name,_i)))
    _x = np.concatenate(_x)
    if _feature_name == 'pmt_fht':
        _x[_x == 1250] = 0
    print(_feature_name + ' has been loaded ')
    return _x


def get_stacked_datanorm(_folder_path):
    _file_num = 1000
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/x_pmt_all_%i.npy' % (_folder_path, _i)):
           temp=np.load('%s/x_pmt_all_%i.npy' % (_folder_path, _i))[:,:,[0,1,3,4,6,2]]
    # fht2, slope, npe, nperatioi5, npemax, peaktime2, timemax
    #        _x.append(temp)
    # _x = np.concatenate(_x)
    # print( 'x_all has been loaded ')
    # return _x
           # Filter the first column (fht2) data
           temp_fht2 = temp[:,:,0]  # Extract the first feature (fht2)
           temp_fht2[temp_fht2 >= 900] = 0
           temp_fht2[temp_fht2 < 0] = 0
           temp[:,:,0] = temp_fht2  # Put the filtered data back
           _x.append(temp)
    _x = np.concatenate(_x)
    print('x_all has been loaded ')
    return _x


def get_stacked_dataweiCNN(_folder_path, _feature_name):
    _file_num = 985
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/x_%s_pmt_%i.npy' % (_folder_path,  _feature_name, _i)):
           _x.append(np.load('%s/x_%s_pmt_%i.npy' % (_folder_path, _feature_name,_i)))
    _x = np.concatenate(_x)
    if _feature_name == 'fht':
        _x[_x == 1250] = 0
    print(_feature_name + ' has been loaded ')
    print(_x.shape)
    return _x