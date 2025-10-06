import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1,2,3'
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from data_utils.PMTLoader import PMTDataLoader,CustomDataset,get_stacked_datanorm,get_stacked_datawei,get_stacked_datachy,get_stacked_dataweiCNN
from data_utils.PMTLoader import jitter_point_cloud, random_point_dropout
# from models.pointnet_regression import get_model,get_loss
from models.pointnet_regression_ssg import get_model,get_loss
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=80, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--data_source',type=str,default='cnn',help='choose from: det|elec|cnn|rawnet')

    parser.add_argument('--log_dir', type=str, default='/disk_pool1/houyh/experiments/test47', help='experiment root')
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
    if isinstance(batch_data, torch.Tensor):
        is_tensor = True
        device = batch_data.device
        batch_data = batch_data.cpu().numpy()
    else:
        is_tensor = False
        
    coords = batch_data[:, :, 0:3].copy()
    features = batch_data[:, :, 3:].copy() if batch_data.shape[2] >= 3 else None
    
    # 1. simulate jitter (only coordinates are affected)
    if jitter:
        coords = jitter_point_cloud(coords, sigma=0.005, clip=0.02)

    if features is not None:
        augmented_data = np.concatenate([coords, features], axis=2)
    else:
        augmented_data = coords
        
    # 2. simulate dropout (only coordinates are affected)
    if dropout:
        augmented_data = random_point_dropout(augmented_data, max_dropout_ratio=0.3)
    
    if is_tensor:
        augmented_data = torch.from_numpy(augmented_data).to(device)
        
    return augmented_data

# def load_data(args):
#     coord_all = np.load('/disk_pool1/houyh/coords/norm_coords')

#     #feature_list = ["pmt_fht", "pmt_slope","pmt_nperatio5",'pmt_peaktime',"pmt_timemax", "pmt_npe"]
#     feature_list = ["fht","slope","peak","timemax","nperatio5"]
#     all_features = []
#     folder_path = '/disk_pool1/houyh/data/y'
#     folder_path1 = '/disk_pool1/chenzhx/rebuilt_data/rawnet/pmt_together4'
#     folder_path2 = '/disk_pool1/houyh/data/det_pmt'
#     folder_path3 = '/disk_pool1/houyh/data/reco_pmt'
#     folder_path4 = '/disk_pool1/houyh/data/elec_pmt'
#     for feature in feature_list:
#        all_features.append(get_stacked_dataweiCNN(folder_path3,feature))
#     all_features.append(get_stacked_dataweiCNN(folder_path4,"npe"))
#     x_all = np.stack(all_features ,axis=-1)

#     #x_all=get_stacked_datanorm(folder_path1)

#     epsilon = 1e-8
#     x_all[:,:,1] = x_all[:,:,1]/(x_all[:,:,-1]+ epsilon)
    
#     # scaler = StandardScaler()
#     # for i in range(x_all.shape[-1]):
#     #     scaler.fit(x_all[:, :, i])
#     #     x_all[:, :, i] = scaler.transform(x_all[:, :, i])


#     x_all = np.concatenate([coord_all,x_all],axis=-1)
#     print("shape of x_all:",x_all.shape) 
   
   
#     # Incident angle
#     y_all = get_stacked_datachy(folder_path,"y")
#     coordx_in = np.sin(y_all[:,0])*np.cos(y_all[:,1])
#     coordy_in = np.sin(y_all[:,0])*np.sin(y_all[:,1])
#     coordz_in = np.cos(y_all[:,0])
#     y_all = np.stack((coordx_in,coordy_in,coordz_in), axis=-1)
#     print("shape of y_all:",y_all.shape) 

#     labels = y_all  
#     points = x_all  

#     points_train, points_test, labels_train, labels_test = train_test_split(points, labels, test_size=0.2, random_state=42)

#     # 仅对“特征部分”做标准化：假设坐标在前3列，之后是可标准化的通道
#     feat_start = 3
#     for i in range(feat_start, points_train.shape[-1]):
#         scaler = StandardScaler()
#         scaler.fit(points_train[:, :, i])             
#         points_train[:, :, i] = scaler.transform(points_train[:, :, i])
#         points_test[:,  :, i] = scaler.transform(points_test[:,  :, i])


#     #Normalization
#     vector_norms = np.sqrt(np.sum(labels_train**2, axis=1))
#     max_norm = np.max(vector_norms)
#     labels_train = labels_train / max_norm
#     labels_test = labels_test / max_norm

#     return points_train, points_test, labels_train, labels_test

def load_data(args):
    coord_all = np.load('/disk_pool1/houyh/coords/norm_coords')
    print("shape of coord_all:", coord_all.shape)
    folder_path_y = '/disk_pool1/houyh/data/y'
    folder_path1 = '/disk_pool1/chenzhx/rebuilt_data/rawnet/pmt_together4'  # rawnet
    folder_path2 = '/disk_pool1/houyh/data/det_pmt'                         # det
    folder_path3 = '/disk_pool1/houyh/data/reco_pmt'                        # reco_pmt
    folder_path4 = '/disk_pool1/houyh/data/elec_pmt'                        # elec_pmt

    src = args.data_source.lower()
    all_features = []

    if src == 'det':
        # 1) detector simulation: 
        feature_list = ["pmt_fht", "pmt_slope","pmt_nperatio5","pmt_peaktime","pmt_timemax","pmt_npe"]
        all_features = [get_stacked_datawei(folder_path2, f) for f in feature_list]
        x_all = np.stack(all_features, axis=-1)

    elif src == 'elec':
        # 2) electronics simulation: 
        feature_list = ["fht","slope","peak","timemax","nperatio5", "npe"]
        all_features = [get_stacked_dataweiCNN(folder_path4, f) for f in feature_list]
        x_all = np.stack(all_features, axis=-1)
        coord_all = coord_all[:9850] 

    elif src == 'cnn':
        # 3) CNN : 
        feature_list = ["fht","slope","peak","timemax","nperatio5"]
        all_features = [get_stacked_dataweiCNN(folder_path3, f) for f in feature_list]
        all_features.append(get_stacked_dataweiCNN(folder_path4, "npe"))
        x_all = np.stack(all_features, axis=-1)
        coord_all = coord_all[:9850]   

    elif src == 'rawnet':
        # 4) Rawnet ：
        x_all = get_stacked_datanorm(folder_path1)

    else:
        raise ValueError(f'Unknown data_source: {args.data_source}')

    if src in ('det', 'reco', 'elec') and x_all.shape[-1] >= 2:
        epsilon = 1e-8
        x_all[:, :, 1] = x_all[:, :, 1] / (x_all[:, :, -1] + epsilon)

    x_all = np.concatenate([coord_all, x_all], axis=-1)
    print("shape of x_all:", x_all.shape)

    # Incident angle


    y_all = get_stacked_datachy(folder_path_y, "y")
    if args.data_source.lower() in ('cnn', 'elec'):
        y_all = y_all[:9850]
    coordx_in = np.sin(y_all[:,0])*np.cos(y_all[:,1])
    coordy_in = np.sin(y_all[:,0])*np.sin(y_all[:,1])
    coordz_in = np.cos(y_all[:,0])
    y_all = np.stack((coordx_in,coordy_in,coordz_in), axis=-1)
    print("shape of y_all:", y_all.shape)

    labels = y_all
    points = x_all


    points_train, points_test, labels_train, labels_test = train_test_split(points, labels, test_size=0.2, random_state=42)
    feat_start = 3
    for i in range(feat_start, points_train.shape[-1]):
        scaler = StandardScaler()
        scaler.fit(points_train[:, :, i])
        points_train[:, :, i] = scaler.transform(points_train[:, :, i])
        points_test[:,  :, i] = scaler.transform(points_test[:,  :, i])

    # normalization
    vector_norms = np.sqrt(np.sum(labels_train**2, axis=1))
    max_norm = np.max(vector_norms)
    labels_train = labels_train / max_norm
    labels_test = labels_test / max_norm

    return points_train, points_test, labels_train, labels_test

def draw_learning_curve(train_losses, test_losses):
    plt.figure(figsize=(12, 8))  
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
    plt.scatter(x, y, label="predictions", color='red', s=10) 
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
    differences = (pred_vals - true_vals) * 180 / np.pi 
    plt.hist(differences, bins=50, alpha=0.7, density=True,
             label='Prediction Errors', color='blue')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
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
    # 计算夹角
    cos_angles = np.sum(true_vals * pred_vals, axis=1) / (
        np.linalg.norm(true_vals, axis=1) * np.linalg.norm(pred_vals, axis=1)
    )
    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # 避免浮点误差导致超出范围
    angles_deg = np.rad2deg(np.arccos(cos_angles))

    # 计算统计量
    (mu, sigma) = norm.fit(angles_deg)
    sorted_angles = np.sort(angles_deg)
    size = len(sorted_angles)
    quantile68 = sorted_angles[int(0.68 * size) - 1] if size > 0 else float('nan')

    # 绘制直方图
    plt.figure()
    plt.hist(angles_deg, bins=100, range=(0, 18), color='green', density=True)
    plt.axvline(x=quantile68, color='black', linewidth=2, linestyle='--', label=f'68% quantile: {quantile68:.2f}')
    plt.xticks(np.arange(0, 19, 6))
    plt.xlim(0, 18)
    plt.legend(frameon=False)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('Opening Angle α ($\degree$)', fontweight='bold')
    plt.ylabel('P.D.F', fontweight='bold')
    plt.title('Opening Angle α P.D.F', fontweight='bold')
    plt.savefig(os.path.join(args.log_dir, 'angel_distribution.png') if args.log_dir else 'angel_distribution.png', dpi=500)
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


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for points, target in tqdm(train_loader,desc='Training'):
        points = points.float().to(device)
        target = target.float().to(device)

        # data augmentation
        #points = augment_point_cloud(points, jitter=True, dropout=True)

        optimizer.zero_grad()
        pred, _ = model(points)

        loss = criterion(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
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
            # data augmentation
            #points = augment_point_cloud(points, jitter=True, dropout=True)

            pred, _ = model(points)
            loss = criterion(pred, target)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return total_loss / len(test_loader), all_preds, all_targets

def main(args):
    logger = setup_logging()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    # Load data
    points_train, points_test, labels_train, labels_test = load_data(args)

    # Modified model initialization
    model = get_model(3, normal_channel=True)
    model = model.to(device)  
    model = model.float()     

    logger.info("Model Architecture:")
    logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    torch.cuda.empty_cache() 
    
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
    # Replace the existing scheduler with one of the following:
    #1. ExponentialLR
    # scheduler = optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=0.95,  # This is equivalent to multiplying by 0.98 each epoch
    # )
    #2. CosineAnnealingLR
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

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
        
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            best_preds = current_preds.copy()
            best_targets = current_targets.copy()
            
        logger.info(f'Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(args.log_dir, 'best_pointnet_regression_model.pth'))
    draw_learning_curve(train_losses, test_losses)

    # draw performance and distribution plots
    theta_predict_test = np.arctan(np.sqrt(np.power(best_preds[:,0], 2) + np.power(best_preds[:,1], 2))/best_preds[:,2])
    theta_true_test = np.arctan(np.sqrt(np.power(best_targets[:,0], 2) + np.power(best_targets[:,1], 2))/best_targets[:,2])
    theta_predict_test[theta_predict_test<0] = theta_predict_test[theta_predict_test<0] + np.pi
    theta_true_test[theta_true_test<0] = theta_true_test[theta_true_test<0] + np.pi
    
    draw_performance(theta_true_test, theta_predict_test)
    
    draw_error_distribution(theta_true_test, theta_predict_test)

    draw_angel_distribution(best_targets, best_preds)
    
    #draw_loss_distribution(torch.tensor(best_preds), torch.tensor(best_targets))

if __name__ == '__main__':
    args = parse_args()
    main(args)