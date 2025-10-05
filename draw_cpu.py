import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from torch.utils.data import DataLoader
from models.pointnet_regression_ssg import get_model
from pointnet_regression_app.tc_pmt_v1 import CustomDataset, load_data

def parse_args():
    parser = argparse.ArgumentParser('Visualization Tool for PointNet Regression Model')
    parser.add_argument('--model_path', type=str, default='./experiments/test35/best_pointnet_regression_model.pth', help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default='/data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./experiments/test35/visualization_results', help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for inference')
    return parser.parse_args()

def load_model_cpu_only(model_path):
    print(f"Loading model from {model_path} (CPU only mode)...")
    device = torch.device("cpu")
    model = get_model(3, normal_channel=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully on CPU")
    return model, device

def process_batch(model, data_loader, device):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (points, target) in enumerate(data_loader):
            # CPU推理
            pred, _ = model(points)
            
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())
            
            # 输出进度
            if batch_idx % 5 == 0:
                print(f"Processed {batch_idx}/{len(data_loader)} batches")
    
    return np.concatenate(all_preds), np.concatenate(all_targets)

def visualize_performance(true_vals, pred_vals, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.scatter(true_vals, pred_vals, label="Predictions", color='red', s=15, alpha=0.5)
    plt.plot([np.amin(true_vals), np.amax(true_vals)], 
             [np.amin(true_vals), np.amax(true_vals)], 
             'k-', alpha=0.75, zorder=0, label="y=x", linewidth=2)
    
    # 计算R²值
    correlation_matrix = np.corrcoef(true_vals, pred_vals)
    r_squared = correlation_matrix[0,1]**2
    
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=16,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(fontsize=16)
    plt.xlabel("True Values", fontsize=20)
    plt.ylabel("Predicted Values", fontsize=20)
    plt.title('Test Performance', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(output_dir, 'performance_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to {os.path.join(output_dir, 'performance_plot.png')}")

def visualize_error_distribution(true_vals, pred_vals, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to {os.path.join(output_dir, 'error_distribution.png')}")

def main():
    args = parse_args()
    print(f"Starting CPU-only visualization process...")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型 (纯CPU模式)
    try:
        # 避免任何GPU加载
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        model, device = load_model_cpu_only(args.model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. 加载测试数据
    try:
        print(f"Loading data from {args.data_path}")
        _, points_test, _, labels_test = load_data(args.data_path)
        print(f"Test data loaded: {points_test.shape}")
        
        test_dataset = CustomDataset(points_test, labels_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0  # 禁用多进程数据加载
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. 进行预测
    try:
        print("Processing test data...")
        predictions, targets = process_batch(model, test_loader, device)
        print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 4. 计算角度
    try:
        theta_predict = np.arctan(np.sqrt(np.power(predictions[:,0], 2) + 
                                        np.power(predictions[:,1], 2))/predictions[:,2])
        theta_true = np.arctan(np.sqrt(np.power(targets[:,0], 2) + 
                                      np.power(targets[:,1], 2))/targets[:,2])
        
        # 调整角度范围
        theta_predict[theta_predict<0] += np.pi
        theta_true[theta_true<0] += np.pi
        
        print(f"Angle calculation completed.")
    except Exception as e:
        print(f"Error during angle calculation: {e}")
        return

    # 5. 生成可视化
    try:
        print("Generating visualizations...")
        visualize_performance(theta_true, theta_predict, args.output_dir)
        visualize_error_distribution(theta_true, theta_predict, args.output_dir)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()