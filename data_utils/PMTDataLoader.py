import os
import numpy as np
import warnings
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

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
    print(_x.shape)
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

# def get_stacked_dataweiCNN(_folder_path, _feature_name):
#     _file_num = 985
#     _x = []
#     for _i in range(_file_num):
#         if os.path.exists('%s/x_%s_pmt_%i.npy' % (_folder_path,  _feature_name, _i)):
#            _x.append(np.load('%s/x_%s_pmt_%i.npy' % (_folder_path, _feature_name,_i)))
#     _x = np.concatenate(_x)
#     if _feature_name == 'fht':
#         _x[_x == 1250] = 0
#     print(_feature_name + ' has been loaded ')
#     print(_x.shape)
#     return _x

def get_stacked_dataweiCNN(_folder_path, _feature_name):
    _file_num = 985  # 限制文件数量
    _x = []
    for _i in range(_file_num):
        file_path = '%s/x_%s_pmt_%i.npy' % (_folder_path, _feature_name, _i)
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(f"Shape of data in file {file_path}: {data.shape}")  # 打印每个文件的数据形状
            _x.append(data)
    _x = np.concatenate(_x)
    if _feature_name == 'fht':
        _x[_x == 1250] = 0
    print(_feature_name + ' has been loaded ')
    print(_x.shape)
    return _x