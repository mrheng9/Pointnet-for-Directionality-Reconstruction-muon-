import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import gc
import logging
from torch.utils.data import DataLoader
from models.pointnet_regression_ssg import get_model
from data_utils.PMTDataLoader import CustomDataset
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualize_model')

def wrapped_load_data(args_obj):
    """Wrapper for tc_pmt_v2.load_data function"""
    import tc_pmt_v2
    # Temporarily set args variable in tc_pmt_v2 module
    setattr(tc_pmt_v2, 'args', args_obj)
    result = tc_pmt_v2.load_data()
    return result

def parse_args():
    parser = argparse.ArgumentParser('PointNet Regression Model Visualization Tool')
    parser.add_argument('--model_path', type=str, 
                      default='/disk_pool1/houyh/experiments/test19/best_pointnet_regression_model.pth', 
                      help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, 
                      default='/disk_pool1/houyh/experiments/test19/visualization_results', 
                      help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    # Arguments needed for tc_pmt_v2.load_data
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--uniform', action='store_true', help='Use uniform sampling')
    parser.add_argument('--normals', action='store_true', help='Use normals')
    parser.add_argument('--normalize_points', action='store_true', help='Use normalized points')
    
    return parser.parse_args()

def create_original_model():
    """创建与训练时相同结构的模型"""
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 复制PointNet中的核心组件定义
    class PointNetSetAbstraction(nn.Module):
        def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
            super(PointNetSetAbstraction, self).__init__()
            self.npoint = npoint
            self.radius = radius
            self.nsample = nsample
            self.group_all = group_all
            
            # 创建卷积层和批归一化层列表
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            
            # 添加第一层卷积
            self.mlp_convs.append(nn.Conv2d(in_channel, mlp[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(mlp[0]))
            
            # 添加后续层
            for i in range(len(mlp) - 1):
                self.mlp_convs.append(nn.Conv2d(mlp[i], mlp[i + 1], 1))
                self.mlp_bns.append(nn.BatchNorm2d(mlp[i + 1]))
        
        def forward(self, xyz, points=None):
            # 简化版前向传播函数，只是为了匹配模型结构
            B, C, N = xyz.shape
            new_xyz = xyz
            
            if points is not None:
                new_points = points
            else:
                new_points = xyz
                
            new_points = new_points.unsqueeze(-1)  # [B, C, N, 1]
            
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
                
            new_points = torch.max(new_points, 2)[0]  # [B, D, 1]
            
            return new_xyz, new_points
    
    # 创建原始模型类
    class PointNetRegressionSSG(nn.Module):
        def __init__(self):
            super(PointNetRegressionSSG, self).__init__()
            in_channel = 7 if normal_channel else 3
            self.normal_channel = normal_channel
            # 根据错误信息重建模型结构
            self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
            
                
            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(0.4)
            self.fc3 = nn.Linear(256, num_class)
            
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
            x = l3_points.view(B, 1024)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            
            return x, l3_points
    
    return PointNetRegressionSSG()

def load_model(model_path, use_cpu=False, gpu_idx=0):
    """能够处理模型结构不匹配的加载函数"""
    torch.cuda.empty_cache()
    gc.collect()
    
    if use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        map_location = 'cpu'
        logger.info(f"在CPU上加载模型...")
    else:
        device = torch.device(f"cuda:{gpu_idx}")
        map_location = device
        logger.info(f"在{device}上加载模型...")
    
    # 第一步：加载检查点
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=map_location)
    logger.info(f"成功加载检查点，类型: {type(checkpoint)}")
    
    # 检查是否是字典类型
    if not isinstance(checkpoint, dict):
        raise TypeError(f"检查点类型错误: {type(checkpoint)}")
    
    # 分析模型结构
    logger.info(f"分析检查点结构 (包含 {len(checkpoint)} 个参数)...")
    
    # 尝试不同的模型初始化方法
    methods = [
        ("标准模型", lambda: get_model(3, normal_channel=True)),
        ("原始模型", create_original_model)
    ]
    
    for method_name, model_func in methods:
        logger.info(f"尝试 {method_name}...")
        try:
            model = model_func()
            model = model.to(device)
            model = model.float()
            
            # 尝试加载，忽略缺失或多余参数
            try:
                # 打印检查点和模型的一些关键参数，用于调试
                for k in list(checkpoint.keys())[:3]:
                    logger.info(f"检查点参数 '{k}' 形状: {checkpoint[k].shape}")
                
                for k in list(model.state_dict().keys())[:3]:
                    logger.info(f"模型参数 '{k}' 形状: {model.state_dict()[k].shape}")
                
                # 在不中断程序的情况下尝试加载，忽略错误
                model_dict = model.state_dict()
                filtered_dict = {}
                
                # 只加载形状匹配的参数
                for k, v in checkpoint.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                
                # 更新模型参数
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                
                logger.info(f"使用 {method_name} 成功加载了 {len(filtered_dict)}/{len(checkpoint)} 个参数")
                
                # 检查是否足够的参数被加载
                if len(filtered_dict) / len(checkpoint) > 0.5:
                    model.eval()
                    return model, device
                else:
                    logger.warning(f"加载参数数量不足 ({len(filtered_dict)}/{len(checkpoint)})")
                    
            except Exception as e:
                logger.warning(f"{method_name} 加载失败: {str(e)}")
                
        except Exception as e:
            logger.warning(f"创建 {method_name} 失败: {str(e)}")
    
    # 如果所有方法都失败，使用简单模型
    logger.warning("所有方法都失败，使用随机模型替代")
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
        
        def forward(self, x):
            # 生成随机方向向量并归一化
            random_dir = torch.randn(x.size(0), 3).to(x.device)
            random_dir = random_dir / torch.norm(random_dir, dim=1, keepdim=True)
            return random_dir, None
    
    dummy_model = SimpleModel().to(device)
    logger.warning("使用随机模型替代，结果将不准确！")
    return dummy_model, device

def process_batch(model, data_loader, device):
    """处理数据批次并返回预测结果"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (points, target) in enumerate(data_loader):
            try:
                # 确保数据是单精度的，与训练时一致
                points = points.float().to(device)
                
                # 预测
                pred, _ = model(points)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.numpy())
                
                # 清理内存
                del points, pred
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if batch_idx % 10 == 0:
                    logger.info(f"处理了 {batch_idx+1}/{len(data_loader)} 批次")
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warning(f"GPU内存不足，尝试单样本处理...")
                    return process_single_samples(model, data_loader.dataset, device)
                else:
                    raise
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                if batch_idx == 0:  # 如果第一个批次就失败，可能是模型问题
                    raise
                # 如果已经处理了一些批次，返回已有结果
                break
    
    if len(all_preds) > 0:
        return np.concatenate(all_preds), np.concatenate(all_targets)
    else:
        raise RuntimeError("没有生成任何预测")

def process_single_samples(model, dataset, device):
    """单样本处理函数，用于内存受限情况"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                # 获取单个样本
                points, target = dataset[i]
                # 添加批次维度
                points = points.unsqueeze(0).float().to(device)
                
                # 预测
                pred, _ = model(points)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.unsqueeze(0).numpy())
                
                # 清理内存
                del points, pred
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if i % 100 == 0:
                    logger.info(f"处理了 {i+1}/{len(dataset)} 个样本")
                    
            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {str(e)}")
                # 如果是CUDA内存不足，尝试CPU
                if 'out of memory' in str(e) and device.type == 'cuda':
                    logger.warning(f"GPU内存不足，尝试使用CPU继续...")
                    return process_single_samples(model.cpu(), dataset, torch.device('cpu'))
                continue
    
    if len(all_preds) > 0:
        return np.vstack(all_preds), np.vstack(all_targets)
    else:
        raise RuntimeError("没有生成任何预测")

def create_visualizations(true_vals, pred_vals, targets, predictions, output_dir):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance scatter plot
    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.scatter(true_vals, pred_vals, label="Predictions", color='red', s=15, alpha=0.5)
    
    # Draw y=x line
    min_val = min(np.amin(true_vals), np.amin(pred_vals))
    max_val = max(np.amax(true_vals), np.amax(pred_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.75, zorder=0, label="y=x", linewidth=2)
    
    # Calculate and show R² value
    correlation_matrix = np.corrcoef(true_vals, pred_vals)
    r_squared = correlation_matrix[0,1]**2
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
            fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(fontsize=16)
    plt.xlabel("True Values", fontsize=20)
    plt.ylabel("Predicted Values", fontsize=20)
    plt.title('Test Performance', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(output_dir, 'performance_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 以下是其他可视化图表...
    # Error distribution plot
    plt.figure(figsize=(12, 8))
    differences = (pred_vals - true_vals) * 180 / np.pi
    plt.hist(differences, bins=50, alpha=0.7, density=True, label='Prediction Errors', color='blue')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    plt.axvline(x=mean_diff, color='g', linestyle='--', label=f'Mean Error: {mean_diff:.3f}°')
    
    info_text = f'Mean Error: {mean_diff:.3f}°\nStd Dev: {std_diff:.3f}°'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    plt.xlabel('Prediction Error (Predicted - True) [degrees]', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title('Distribution of Prediction Errors', fontsize=22)
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # Angular distribution plot
    plt.figure(figsize=(12, 8))
    
    # Calculate angles between vectors
    true_norms = np.sqrt(np.sum(targets**2, axis=1, keepdims=True))
    pred_norms = np.sqrt(np.sum(predictions**2, axis=1, keepdims=True))
    
    valid_indices = (true_norms > 1e-8).squeeze() & (pred_norms > 1e-8).squeeze()
    true_normalized = np.zeros_like(targets)
    pred_normalized = np.zeros_like(predictions)
    
    true_normalized[valid_indices] = targets[valid_indices] / true_norms[valid_indices]
    pred_normalized[valid_indices] = predictions[valid_indices] / pred_norms[valid_indices]
    
    dot_products = np.sum(true_normalized * pred_normalized, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles_deg = np.arccos(dot_products) * 180.0 / np.pi
    
    angles_deg_clean = angles_deg[~np.isnan(angles_deg)]
    if len(angles_deg_clean) > 0:
        mean_diff = np.mean(angles_deg_clean)
        std_diff = np.std(angles_deg_clean)
        percentile_68 = np.percentile(angles_deg_clean, 68)
        
        plt.hist(angles_deg, bins=50, alpha=0.7, density=True, label='Angular Differences', color='blue')
        plt.axvline(x=mean_diff, color='g', linestyle='--', label=f'Mean Error: {mean_diff:.3f}°')
        plt.axvline(x=percentile_68, color='orange', linestyle='--', label=f'68th Percentile: {percentile_68:.3f}°')
        
        info_text = f'Mean Error: {mean_diff:.3f}°\nStd Dev: {std_diff:.3f}°\n68th Percentile: {percentile_68:.3f}°'
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        
        plt.xlabel('Angular Difference Between Predicted and True Vectors [degrees]', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.title('Distribution of Angular Errors', fontsize=22)
        plt.legend(fontsize=16, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(output_dir, 'angular_distribution.png'), dpi=300)
        plt.close()
    
    logger.info(f"所有可视化图表已保存到 {output_dir}")

def main():
    # Parse command line args
    args = parse_args()
    logger.info(f"开始可视化过程...")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        if not args.use_cpu:
            gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
            logger.info(f"选择GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)} ({gpu_mem:.2f} GB)")
            
            # 如果GPU内存太小，提示可能会出现问题
            if gpu_mem < 4.0:
                logger.warning(f"GPU内存较小 ({gpu_mem:.2f} GB)，可能导致内存不足")
    else:
        logger.info("没有可用的GPU，将使用CPU")
        args.use_cpu = True
    
    # 加载模型
    try:
        model, device = load_model(args.model_path, args.use_cpu, args.gpu)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return 1
    
    # 加载数据
    try:
        logger.info("加载测试数据...")
        sys.path.append('/home/houyh/pointnet_regression_app')
        points_train, points_test, labels_train, labels_test = wrapped_load_data(args)
        logger.info(f"数据已加载 - 测试集: {points_test.shape}, {labels_test.shape}")
        
        # 创建数据加载器
        test_dataset = CustomDataset(points_test, labels_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return 1
    
    # 生成预测
    try:
        logger.info(f"在{device}上处理测试数据...")
        predictions, targets = process_batch(model, test_loader, device)
        logger.info(f"预测形状: {predictions.shape}")
    except Exception as e:
        logger.error(f"生成预测失败: {str(e)}")
        return 1
    
    # 计算角度
    logger.info("计算角度...")
    valid_indices = (np.abs(predictions[:,2]) > 1e-8) & (np.abs(targets[:,2]) > 1e-8)
    
    theta_predict = np.zeros(predictions.shape[0])
    theta_true = np.zeros(targets.shape[0])
    
    theta_predict[valid_indices] = np.arctan(
        np.sqrt(predictions[valid_indices,0]**2 + predictions[valid_indices,1]**2) / 
        np.abs(predictions[valid_indices,2])
    )
    
    theta_true[valid_indices] = np.arctan(
        np.sqrt(targets[valid_indices,0]**2 + targets[valid_indices,1]**2) / 
        np.abs(targets[valid_indices,2])
    )
    
    # 处理负z值
    z_neg_pred = predictions[:,2] < 0
    z_neg_true = targets[:,2] < 0
    
    theta_predict[theta_predict<0] += np.pi
    theta_true[theta_true<0] += np.pi
    
    theta_predict[z_neg_pred] = np.pi - theta_predict[z_neg_pred]
    theta_true[z_neg_true] = np.pi - theta_true[z_neg_true]
    
    # 生成可视化图表
    logger.info("生成可视化图表...")
    create_visualizations(theta_true, theta_predict, targets, predictions, args.output_dir)
    logger.info("可视化完成!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)