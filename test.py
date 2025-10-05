import os
import numpy as np

# 文件夹路径
folder_path4 = '/disk_pool1/houyh/data/elec_pmt'

# 检查文件数量
files = os.listdir(folder_path4)
print(f"Total files in folder_path4: {len(files)}")

# 检查文件命名规则
print("File names in folder_path4:")
for file in files[:10]:  # 仅打印前10个文件名
    print(file)

# 定义加载函数
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

# 加载 npe 数据并验证形状
npe = get_stacked_dataweiCNN(folder_path4, "npe")
print(f"Shape of npe: {npe.shape}")

coord_data = np.load('/disk_pool1/houyh/coords/norm_coords')
print(f"Shape of coord_data: {coord_data.shape}")