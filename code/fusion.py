import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from torch.cuda.amp import GradScaler, autocast
    
# 设置随机种子
SEED = 34
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# RNA二级结构特征提取器
class RNAStructureFeatureExtractor:
    @staticmethod
    def extract_basic_features(structure):
        return {
            'struct_length': len(structure),
            'paired_bases': structure.count('(') + structure.count(')'),
            'unpaired_bases': structure.count('.'),
            'hairpin_loops': structure.count('()'),
            'dot_ratio': structure.count('.') / max(1, len(structure))
        }
    
    @staticmethod
    def calculate_depth_features(structure):
        depth = 0
        max_depth = 0
        depth_sequence = []
        
        for char in structure:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
            depth_sequence.append(depth)
        
        return {
            'max_depth': max_depth,
            'mean_depth': np.mean(depth_sequence),
            'depth_variance': np.var(depth_sequence)
        }
    
    @staticmethod
    def count_structural_elements(structure):
        return {
            'long_hairpins': len(re.findall(r'\(\.{3,}\)', structure)),
            'nested_stems': len(re.findall(r'\(+\.+\)+', structure))
        }
    
    @staticmethod
    def extract_all_features(structure):
        features = {
            **RNAStructureFeatureExtractor.extract_basic_features(structure),
            **RNAStructureFeatureExtractor.calculate_depth_features(structure),
            **RNAStructureFeatureExtractor.count_structural_elements(structure)
        }
        return features

# 数据集类
class RNADataset(Dataset):
    def __init__(self, pos_ohnd_file, pos_struct_file, neg_ohnd_file, neg_struct_file):
        # 加载OH+ND特征
        pos_ohnd = pd.read_csv(pos_ohnd_file).iloc[:, 1:206].values.astype(np.float32)
        neg_ohnd = pd.read_csv(neg_ohnd_file).iloc[:, 1:206].values.astype(np.float32)
        
        # 加载结构特征并处理二级结构
        pos_struct_df = pd.read_csv(pos_struct_file)
        neg_struct_df = pd.read_csv(neg_struct_file)

        # 提取二级结构特征
        pos_struct_features = np.array([
            list(RNAStructureFeatureExtractor.extract_all_features(s).values())
            for s in pos_struct_df['secondary_structure']
        ], dtype=np.float32)
        
        neg_struct_features = np.array([
            list(RNAStructureFeatureExtractor.extract_all_features(s).values())
            for s in neg_struct_df['secondary_structure']
        ], dtype=np.float32)
        
        # 提取其他数值特征（跳过sequence和secondary_structure列）
        pos_other_features = pos_struct_df.drop(['sequence', 'secondary_structure'], axis=1).values.astype(np.float32)
        neg_other_features = neg_struct_df.drop(['sequence', 'secondary_structure'], axis=1).values.astype(np.float32)

        # 删除指定的列（第7、9、11、13列，假设是1-based索引）
        columns_to_drop = [6, 8, 10, 12]  # 转换为0-based索引
        pos_other_features = np.delete(pos_other_features, columns_to_drop, axis=1)
        neg_other_features = np.delete(neg_other_features, columns_to_drop, axis=1)
        
        # 合并结构特征
        pos_struct = np.hstack([pos_other_features, pos_struct_features])
        neg_struct = np.hstack([neg_other_features, neg_struct_features])

        # 合并正负样本
        self.ohnd_features = np.vstack([pos_ohnd, neg_ohnd])
        self.structural_features = np.vstack([pos_struct, neg_struct])
        self.labels = np.concatenate([np.ones(len(pos_ohnd)), np.zeros(len(neg_ohnd))])
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.structural_features = self.scaler.fit_transform(self.structural_features)

        self._shuffle_data()
        self.training = False
    
    def _shuffle_data(self):
        indices = np.random.permutation(len(self.ohnd_features))
        self.ohnd_features = self.ohnd_features[indices]
        self.structural_features = self.structural_features[indices]
        self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.ohnd_features)
    
    def _augment_data(self, ohnd, structural):
        # 对OHND特征添加轻微噪声
        if random.random() < 0.5:
            noise = torch.randn_like(ohnd) * 0.01
            ohnd = ohnd + noise

        # 对结构特征进行随机掩码
        if random.random() < 0.3:
            mask = torch.rand_like(structural) < 0.1
            structural = structural * (~mask).float()

        return ohnd, structural

    def set_training_mode(self, mode):
        """设置数据集模式（训练/测试）"""
        self.training = mode

    def __getitem__(self, idx):
        ohnd = self.ohnd_features[idx].reshape(41, 5).transpose()
        structural = self.structural_features[idx]

        # 只在训练时增强数据
        if self.training and random.random() < 0.5:  # 50%概率增强
            # 添加轻微噪声
            ohnd = ohnd + np.random.normal(0, 0.01, size=ohnd.shape).astype(np.float32)
            # 随机掩码部分结构特征
            mask = np.random.rand(*structural.shape) < 0.1
            structural = structural * (1 - mask.astype(np.float32))

        return (
            torch.FloatTensor(ohnd),
            torch.FloatTensor(structural),
            torch.LongTensor([self.labels[idx]])
        )

def collate_fn(batch):
        """自定义批次处理函数"""
        ohnd = torch.stack([item[0] for item in batch])
        structural = torch.stack([item[1] for item in batch])
        labels = torch.cat([item[2] for item in batch])
        return ohnd, structural, labels


def create_loaders(dataset, batch_size=256, test_ratio=0.2):
    """创建数据加载器"""
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=dataset.labels,
        random_state=SEED
    )

    # 创建训练集子集并设置模式
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_subset.dataset.set_training_mode(True)  # 设置训练模式
    
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # 创建测试集子集并设置模式
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_subset.dataset.set_training_mode(False)  # 设置测试模式

    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, test_loader


# 模型组件（保持不变）
class ConvFactory(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = self.bn(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, layers, growth_rate, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(ConvFactory(in_channels + i * growth_rate, growth_rate, dropout_rate))
    
    def forward(self, x):
        feature_maps = [x]
        for layer in self.layers:
            x = layer(torch.cat(feature_maps, dim=1))
            feature_maps.append(x)
        return torch.cat(feature_maps, dim=1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1))))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class StructuralTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32))
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        x = x.flatten(1)
        return self.classifier(x)

# 主模型
class CreateModel(nn.Module):
    def __init__(self, input_channels=5, structural_features=18,
                 denseblocks=4, layers=3, filters=96, growth_rate=32, 
                 dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 序列特征分支
        self.initial_conv = nn.Conv1d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)
        
        # 稠密块
        self.denseblocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i in range(denseblocks - 1):
            self.denseblocks.append(DenseBlock(filters, layers, growth_rate, dropout_rate))
            filters += layers * growth_rate
            self.transitions.append(Transition(filters, filters // 2, dropout_rate))
            filters = filters // 2
        
        self.denseblocks.append(DenseBlock(filters, layers, growth_rate, dropout_rate))
        filters += layers * growth_rate
        
        # 注意力机制
        self.cbam = CBAM(filters)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 结构特征分支
        self.structural_transformer = StructuralTransformer(
            input_dim=structural_features,
            embed_dim=64,
            num_heads=8,
            num_layers=2,
            dropout=dropout_rate)
        
        # 特征融合
        self.gate = nn.Sequential(
            nn.Linear(filters + 32, 256),
            nn.GELU(),
            nn.Linear(256, filters),
            nn.Sigmoid())
        
        # 可学习的加权融合
        self.structural_proj = nn.Linear(32, filters)  # 将结构特征投影到与序列特征相同维度
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 添加更多的Dropout层
        self.final_dropout = nn.Dropout(0.2)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(filters + 32, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 2))
        self.classifier1 = nn.Sequential(
            nn.Linear(filters, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 2))

    
    def forward(self, x_sequence, x_structural):
        # 序列特征处理
        x_sequence = self.initial_conv(x_sequence)
        x_sequence = self.bn(x_sequence)
        x_sequence = self.relu(x_sequence)

        for block, transition in zip(self.denseblocks[:-1], self.transitions):
            x_sequence = block(x_sequence)
            x_sequence = transition(x_sequence)

        x_sequence = self.denseblocks[-1](x_sequence)
        x_sequence = self.cbam(x_sequence)
        x_sequence = self.avg_pool(x_sequence).squeeze(-1)

        # 结构特征处理
        structural_features = self.structural_transformer(x_structural)

        
        # 1.特征融合
        combined = torch.cat([x_sequence, structural_features], dim=1)
        gate = self.gate(combined)
        gated_sequence = x_sequence * gate
        final_feature = torch.cat([gated_sequence, structural_features], dim=1)

        # # 2.替换门控融合为简单连接
        # final_feature = torch.cat([x_sequence, structural_features], dim=1)

        # # 3.使用可学习的权重进行加权融合
        # structural_features = self.structural_proj(structural_features)  # 投影到 [batch, filters]
        # final_feature = self.alpha * x_sequence + (1 - self.alpha) * structural_features
        # final_feature = self.final_dropout(final_feature)
        # return self.classifier1(final_feature)

        # 在分类器前添加dropout
        final_feature = self.final_dropout(final_feature)
        return self.classifier(final_feature)

def train_model(model, train_loader, test_loader, criterion, optimizer, device,
               num_epochs=50, patience=5, min_delta=0.001, model_path='best_model.pth'):
    best_acc = 0.0
    best_metrics = {}
    best_model_state = None
    no_improve_epochs = 0
    history = {
        'train_loss': [],
        'test_sn': [],
        'test_sp': [],
        'test_acc': [],
        'test_f1': [],
        'test_mcc': [],
        'test_auroc': []
    }

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs_conv, inputs_domain, labels in train_loader:
            inputs_conv = inputs_conv.to(device)
            inputs_domain = inputs_domain.to(device)
            labels = labels.squeeze().to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs_conv, inputs_domain)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练指标
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        
        # 测试阶段（使用原有test_model函数）
        current_sn, current_sp, current_acc, current_f1, current_mcc, current_auroc = test_model(model, test_loader, device, verbose=False)
        history['test_sn'].append(current_sn)
        history['test_sp'].append(current_sp)
        history['test_acc'].append(current_acc)
        history['test_f1'].append(current_f1)
        history['test_mcc'].append(current_mcc)
        history['test_auroc'].append(current_auroc)

        # 原有早停逻辑
        if current_acc - best_acc > min_delta:
            best_acc = current_acc
            best_metrics = {
                'sn': current_sn,
                'sp': current_sp,
                'acc': current_acc,
                'f1': current_f1,
                'mcc': current_mcc,
                'auroc': current_auroc
            }
            best_model_state = model.state_dict()
            no_improve_epochs = 0
            torch.save(best_model_state, model_path)
            print(f"Saved best model with Acc: {best_acc:.4f}")
        else:
            no_improve_epochs += 1
        
        # 原有打印格式
        print(f'Epoch {epoch+1}/{num_epochs} - '
            f'Train Loss: {epoch_loss:.4f} | '
            f'Test Acc: {current_acc:.4f} (Best: {best_acc:.4f}) | '
            f'SN: {current_sn:.4f} | SP: {current_sp:.4f} | '
            f'F1: {current_f1:.4f} | MCC: {current_mcc:.4f} | '
            f'AUROC: {current_auroc:.4f} | '
            f'Patience: {no_improve_epochs}/{patience}')
            
        if no_improve_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_metrics

# 原有test_model函数保持不变
def test_model(model, test_loader, device, verbose=True):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs_conv, inputs_domain, labels in test_loader:
            inputs_conv = inputs_conv.to(device)
            inputs_domain = inputs_domain.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(inputs_conv, inputs_domain)
            probs = outputs[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    
    if verbose:
        print(f'Test Metrics - SN: {sn:.4f} | SP: {sp:.4f} | ACC: {acc:.4f} | '
              f'F1: {f1:.4f} | MCC: {mcc:.4f} | AUROC: {auroc:.4f}')
    
    return sn, sp, acc, f1, mcc, auroc


# 主程序
if __name__ == '__main__':
    # 配置
    config = {
        'pos_ohnd': 'seq_fold/data/pos_encoding_OH_ND.csv',
        'pos_fold': 'seq_fold/data/pos_fold.csv',
        'neg_ohnd': 'seq_fold/data/neg_encoding_OH_ND.csv',
        'neg_fold': 'seq_fold/data/neg_fold.csv',
        'batch_size': 128,
        'lr': 0.001,
        'num_epochs': 100,
        'patience': 15,
        'model_path': 'seq_fold/model/GFN_model1.pth'
    }


    # 创建数据集
    full_dataset = RNADataset(
        pos_ohnd_file=config['pos_ohnd'],
        pos_struct_file=config['pos_fold'],
        neg_ohnd_file=config['neg_ohnd'],
        neg_struct_file=config['neg_fold']
    )
    
    # 创建数据加载器
    train_loader, test_loader = create_loaders(full_dataset, batch_size=config['batch_size'])
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CreateModel(structural_features=full_dataset.structural_features.shape[1]).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # 训练模型
    history, best_metrics = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        model_path=config['model_path']
    )
    print("\nFinal Test Results:")
    model.load_state_dict(torch.load(config['model_path']))
    # 最终测试
    
    test_model(model, test_loader, device)