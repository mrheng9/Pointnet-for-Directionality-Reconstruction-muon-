import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from data_utils.PMTDataLoader import CustomDataset,get_stacked_datach, get_stacked_datanorm,get_stacked_datawei,get_stacked_datachy,get_stacked_dataweiCNN
from models.pointnet_regression_radius import get_model,get_loss
import provider 
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import provider  

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='3,2,1', help='specify gpu device')

    parser.add_argument('--epoch', default=80, type=int, help='number of epoch in training')    
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')

    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--uniform', default=True, help='Use uniform sampling')
    parser.add_argument('--normals',default=True, help='Use normals')
    parser.add_argument('--normalize_points', default=True, help='Use normalized points')
    parser.add_argument('--process_data', default=True, help='Process data')   
    
    parser.add_argument('--log_dir', type=str, default='/disk_pool1/houyh/experiments/test25', help='experiment root')
    return parser.parse_args()
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def augment_point_cloud(batch_data, jitter=True, dropout=True):
    # 将tensor转换为numpy进行处理
    if isinstance(batch_data, torch.Tensor):
        is_tensor = True
        device = batch_data.device
        batch_data = batch_data.cpu().numpy()
    else:
        is_tensor = False
        
    # 仅对坐标部分进行增强（前3个通道）
    coords = batch_data[:, :, 0:3].copy()
    features = batch_data[:, :, 3:].copy() if batch_data.shape[2] >= 3 else None
    
    # 1. 添加抖动模拟测量误差（仅应用于坐标）
    if jitter:
        coords = provider.jitter_point_cloud(coords, sigma=0.005, clip=0.02)
    
    # 组合坐标和特征
    if features is not None:
        augmented_data = np.concatenate([coords, features], axis=2)
    else:
        augmented_data = coords
        
    # 2. 模拟点丢失（坐标和特征都受影响）
    if dropout:
        augmented_data = provider.random_point_dropout(augmented_data, max_dropout_ratio=0.3)
    
    # 将numpy数组转回tensor（如果输入是tensor）
    if is_tensor:
        augmented_data = torch.from_numpy(augmented_data).to(device)
        
    return augmented_data

def normalize_to_unit_vector(vectors):
    # 计算向量长度
    norms = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    # 归一化为单位向量
    return vectors / norms
def load_data():
    # Load preprocessed coordinates and sample indices
    coords = np.load('/home/houyh/pointnet_regression_app/data/processed_coords%s.npy'%(args.num_point))
    sample_indices = np.load('/home/houyh/pointnet_regression_app/data/sample_indices%s.npy'%(args.num_point))
    print(sample_indices.shape)

    # feature_list = ["pmt_fht", "pmt_npe", "pmt_slope","pmt_nperatio5"]
    feature_list = ["fht","slope", "nperatio5"]
    all_features = []
    folder_path = '/disk_pool1/houyh/data/y' #6,7,8 y 9,10,11
    # folder_path = '/disk_pool1/chenzhx/random_muon/detsim/y' #2,3,4 y_all 5,6,7
    # folder_path0 = '/disk_pool1/chenzhx/rebuilt_data/CNN'
    folder_path1 = '/disk_pool1/chenzhx/rebuilt_data/rawnet/pmt_together4'
    folder_path2 = '/disk_pool1/houyh/data/det_pmt'
    folder_path3 = '/disk_pool1/houyh/data/reco_pmt'
    folder_path4 = '/disk_pool1/houyh/data/elec_pmt'
    for feature in feature_list:
       all_features.append(get_stacked_dataweiCNN(folder_path3,feature))
    all_features.append(get_stacked_dataweiCNN(folder_path4,"npe"))

    x_all = np.stack(all_features ,axis=-1)
    batch_size = x_all.shape[0]
    coords = coords[:batch_size]
    sample_indices = sample_indices[:batch_size]
    sampled_x_all = np.array([x_all[i, sample_indices[i], :] for i in range(len(sample_indices))])
    x_all = np.concatenate([coords,sampled_x_all],axis=-1)
    print("shape of x_all:",x_all.shape) 

    # x_all=get_stacked_datanorm(folder_path1)    
    # batch_size = x_all.shape[0]
    # coords = coords[:batch_size]
    # sampled_x_all = np.array([x_all[i, sample_indices[i], :] for i in range(len(sample_indices))])
    # x_all = np.concatenate([coords,sampled_x_all],axis=-1)
    # print("shape of x_all:",x_all.shape) 
   
    #入射方向
    y_all = get_stacked_datachy(folder_path,"y")
    # coordx_in = y_all[:,6]
    # coordy_in = y_all[:,7]
    # coordz_in = y_all[:,8]
    coordx_in = np.sin(y_all[:,0])*np.cos(y_all[:,1])
    coordy_in = np.sin(y_all[:,0])*np.sin(y_all[:,1])
    coordz_in = np.cos(y_all[:,0])
    y_all = np.stack((coordx_in,coordy_in,coordz_in), axis=-1)
    print("shape of y_all:",y_all.shape) 

    labels = y_all  # Adjust this according to your data format
    points = x_all  # Adjust this according to your data format

    # 划分训练集和测试集
    points_train, points_test, labels_train, labels_test = train_test_split(points, labels,test_size=0.2, random_state=42)
    #将前2000个样本作为测试集，其余作为训练集
    # points_test = points[:2000]
    # points_train = points[2000:]
    # labels_test = labels[:2000] 
    # labels_train = labels[2000:]

    # 标准化
    mean = np.mean(labels_train, axis=0)
    std = np.std(labels_train, axis=0)
    labels_train = (labels_train - mean) / std
    labels_test = (labels_test - mean) / std
    # min_max 归一化
    # labels_train = normalize_to_unit_vector(labels_train)
    # labels_test = normalize_to_unit_vector(labels_test)
    return points_train, points_test, labels_train, labels_test

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for points, target in tqdm(train_loader,desc='Training'):
        points = points.float().to(device)
        target = target.float().to(device)

        # 应用数据增强
        #points = augment_point_cloud(points, jitter=True, dropout=True)

        optimizer.zero_grad()
        pred, _ = model(points)

        loss = criterion(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for points, target in tqdm(test_loader, desc='Testing'):
            points = points.to(device)
            target = target.to(device)
            # 应用数据增强
            #points = augment_point_cloud(points, jitter=True, dropout=True)

            pred, _ = model(points)
            loss = criterion(pred, target)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return total_loss / len(test_loader), all_preds, all_targets



def draw_learning_curve(train_losses, test_losses):
    plt.figure(figsize=(12, 8))  # 增加图像整体大小
    plt.plot(range(1, args.epoch + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, args.epoch + 1), test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.legend(fontsize=16)
    plt.title('Learning Curve', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.yscale('log')
    plt.savefig(os.path.join(args.log_dir, 'learning_curve.png') if args.log_dir else 'learning_curve.png', dpi=300)
    plt.close()
def draw_performance(x, y):
    plt.clf()
    # plt.figure(figsize=(12, 8))
    plt.grid()
    plt.scatter(x, y, label="predictions", color='red', s=6) 
    plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x", linewidth=2)
    plt.legend(fontsize=16)
    plt.xlabel("True", fontsize=20)
    plt.ylabel("Predicted", fontsize=20)
    plt.title('Test Performance', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(args.log_dir, 'Test Performance.png') if args.log_dir else 'Test Performance.png', dpi=300)
    plt.close()
def draw_error_distribution(true_vals, pred_vals):
    plt.figure(figsize=(12, 8))
    
    # 计算预测误差 (转换为角度)
    differences = (pred_vals - true_vals) * 180 / np.pi 

    # 绘制误差分布直方图
    plt.hist(differences, bins=50, alpha=0.7, density=True,
             label='Prediction Errors', color='blue')
    
    # 添加零误差线
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    
    # 添加统计信息
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    plt.axvline(x=mean_diff, color='g', linestyle='--',
                label=f'Mean Error')
    
    info_text = f'Mean Error: {mean_diff:.3f}°\nStd Dev: {std_diff:.3f}°'
    plt.text(0.05, 0.95, info_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.xlabel('Prediction Error (Predicted - True) [degrees]', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title('Distribution of Prediction Errors', fontsize=22)
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(args.log_dir, 'error_distribution.png') if args.log_dir else 'error_distribution.png', dpi=300)
    plt.close()
def draw_angel_distribution(true_vals, pred_vals):
    plt.figure(figsize=(12, 8))
    
    # 计算夹角并添加安全措施
    true_norms = np.sqrt(np.sum(true_vals**2, axis=1, keepdims=True))
    pred_norms = np.sqrt(np.sum(pred_vals**2, axis=1, keepdims=True))

    # 避免除以零
    valid_indices = (true_norms > 1e-8).squeeze() & (pred_norms > 1e-8).squeeze()
    true_normalized = np.zeros_like(true_vals)
    pred_normalized = np.zeros_like(pred_vals)

    true_normalized[valid_indices] = true_vals[valid_indices] / true_norms[valid_indices]
    pred_normalized[valid_indices] = pred_vals[valid_indices] / pred_norms[valid_indices]

    dot_products = np.sum(true_normalized * pred_normalized, axis=1)

    # 确保点积在有效范围内
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # 计算角度
    angles_rad = np.arccos(dot_products)
    angles_deg = angles_rad * 180.0 / np.pi
    
    # 先计算统计量，确保无论走哪个分支都有值
    mean_diff = np.mean(angles_deg)
    std_diff = np.std(angles_deg)

    # 过滤掉任何NaN值然后计算分位数
    angles_deg_clean = angles_deg[~np.isnan(angles_deg)]
    if len(angles_deg_clean) > 0:
        percentile_68 = np.percentile(angles_deg_clean, 68)
        plt.axvline(x=percentile_68, color='orange', linestyle='--', 
                    label=f'68th Percentile: {percentile_68:.3f}°')
    else:
        print("Warning: No valid angles to calculate percentile")
        # 没有有效角度时，使用默认值或其他处理方式
        percentile_68 = float('nan')

    # 绘制直方图
    plt.hist(angles_deg, bins=50, alpha=0.7, density=True,
            label='Angular Differences', color='blue')
    
    # 添加零误差线和平均误差线
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.axvline(x=mean_diff, color='g', linestyle='--', label=f'Mean Error')
    
    # 如果之前未定义percentile_68，则重新计算
    if 'percentile_68' not in locals():
        percentile_68 = np.percentile(angles_deg, 68)
    plt.axvline(x=percentile_68, color='orange', linestyle='--', 
                label=f'68th Percentile: {percentile_68:.3f}°')
    
    info_text = f'Mean Error: {mean_diff:.3f}°\nStd Dev: {std_diff:.3f}°\n68th Percentile: {percentile_68:.3f}°'
    plt.text(0.05, 0.95, info_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.xlabel('Angular Difference Between Predicted and True Vectors [degrees]', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title('Distribution of Angular Errors', fontsize=22)
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(args.log_dir, 'angel_distribution.png') if args.log_dir else 'angel_distribution.png', dpi=300)
    plt.close()

def calculate_loss_distribution(pred_vals, true_vals):
    # Calculate the loss using the same method as in get_loss
    losses = torch.sqrt((true_vals[:, 0] - pred_vals[:, 0])**2 + 
                        (true_vals[:, 1] - pred_vals[:, 1])**2 + 
                        (true_vals[:, 2] - pred_vals[:, 2])**2)
    return losses.cpu().numpy()
def draw_loss_distribution(pred_vals, true_vals):
    losses = calculate_loss_distribution(pred_vals, true_vals)
    
    plt.figure(figsize=(12, 8))
    plt.hist(losses, bins=50, alpha=0.7, density=True, color='blue', label='Loss Distribution')
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    plt.axvline(x=mean_loss, color='g', linestyle='--', label=f'Mean Loss: {mean_loss:.3f}')
    
    info_text = f'Mean Loss: {mean_loss:.3f}\nStd Dev: {std_loss:.3f}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    plt.xlabel('Loss', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title('Distribution of Losses', fontsize=22)
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if args.log_dir:
        plt.savefig(os.path.join(args.log_dir, 'loss_distribution.png'), dpi=300)
    else:
        plt.savefig('loss_distribution.png', dpi=300)
    
    plt.show()
    plt.close()


def main(args):
    logger = setup_logging()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    # 加载数据并划分训练集和测试集
    points_train, points_test, labels_train, labels_test = load_data()

    # Modified model initialization
    model = get_model(3, normal_channel=True)
    model = model.to(device)  # Move model to device first
    model = model.float()     # Then convert to float

    # 使用 DataParallel 包装模型以支持多GPU
    if torch.cuda.device_count() > 1 and not args.use_cpu:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    # 打印模型信息
    logger.info("Model Architecture:")
    logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    torch.cuda.empty_cache()  # Clear GPU memory cache
    
    criterion = get_loss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    train_dataset = CustomDataset(points_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CustomDataset(points_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses = []
    test_losses = []

    best_test_loss = float('inf')
    best_model_state = None
    best_preds = None
    best_targets = None

    for epoch in range(args.epoch):
        train_loss = train(model, criterion, optimizer, train_loader, device)
        test_loss, current_preds, current_targets = evaluate(model, criterion, test_loader, device)
        
        scheduler.step(test_loss) # 根据test_loss调整学习率
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            best_preds = current_preds.copy()
            best_targets = current_targets.copy()
            
        logger.info(f'Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # 保存最佳模型
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(args.log_dir, 'best_pointnet_regression_model.pth'))

    # 绘制学习曲线
    draw_learning_curve(train_losses, test_losses)

    # 使用最佳模型的预测结果绘制性能图
    theta_predict_test = np.arctan(np.sqrt(np.power(best_preds[:,0], 2) + np.power(best_preds[:,1], 2))/best_preds[:,2])
    theta_true_test = np.arctan(np.sqrt(np.power(best_targets[:,0], 2) + np.power(best_targets[:,1], 2))/best_targets[:,2])
    theta_predict_test[theta_predict_test<0] = theta_predict_test[theta_predict_test<0] + np.pi
    theta_true_test[theta_true_test<0] = theta_true_test[theta_true_test<0] + np.pi
    
    # 绘制性能散点图
    draw_performance(theta_true_test, theta_predict_test)
    
    # 绘制分布图
    draw_error_distribution(theta_true_test, theta_predict_test)

    # 绘制分布图
    draw_angel_distribution(best_targets, best_preds)
    
    # 绘制损失分布图
    #draw_loss_distribution(torch.tensor(best_preds), torch.tensor(best_targets))

if __name__ == '__main__':
    args = parse_args()
    main(args)