import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
from torch.cuda.amp import GradScaler, autocast
import time
import os

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

class RNADataset(Dataset):
    def __init__(self, pos_ohnd_file, pos_struct_file, neg_ohnd_file, neg_struct_file):
        # 加载seq特征
        pos_ohnd = pd.read_csv(pos_ohnd_file).iloc[:, 1:206].values.astype(np.float32)
        neg_ohnd = pd.read_csv(neg_ohnd_file).iloc[:, 1:206].values.astype(np.float32)
        
        # 加载structural特征
        pos_struct_df = pd.read_csv(pos_struct_file)
        neg_struct_df = pd.read_csv(neg_struct_file)

        pos_struct_features = np.array([
            list(RNAStructureFeatureExtractor.extract_all_features(s).values())
            for s in pos_struct_df['secondary_structure']
        ], dtype=np.float32)
        
        neg_struct_features = np.array([
            list(RNAStructureFeatureExtractor.extract_all_features(s).values())
            for s in neg_struct_df['secondary_structure']
        ], dtype=np.float32)
        
        pos_other_features = pos_struct_df.drop(['sequence', 'secondary_structure'], axis=1).values.astype(np.float32)
        neg_other_features = neg_struct_df.drop(['sequence', 'secondary_structure'], axis=1).values.astype(np.float32)

        pos_struct = np.hstack([pos_other_features, pos_struct_features])
        neg_struct = np.hstack([neg_other_features, neg_struct_features])

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
    
    def set_training_mode(self, mode):
        self.training = mode

    def __getitem__(self, idx):
        ohnd = self.ohnd_features[idx].reshape(41, 5).transpose(1, 0)
        structural = self.structural_features[idx]

        if self.training and random.random() < 0.5:
            # 数据增强
            ohnd = ohnd + np.random.normal(0, 0.01, size=ohnd.shape).astype(np.float32)
            mask = np.random.rand(*structural.shape) < 0.1
            structural = structural * (1 - mask.astype(np.float32))
            
        return (
            torch.FloatTensor(ohnd),
            torch.FloatTensor(structural),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def collate_fn(batch):
    ohnd = torch.stack([item[0] for item in batch])
    structural = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return ohnd, structural, labels

def create_cv_loaders(dataset, batch_size=256, n_splits=5, test_size=0.2):
    """创建交叉验证数据加载器"""
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size,
        stratify=dataset.labels,
        random_state=34
    )
    
    test_subset = Subset(dataset, test_idx)
    test_subset.dataset.set_training_mode(False)
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_loaders = []
    
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(train_idx, dataset.labels[train_idx])):
        train_subset = Subset(dataset, train_idx[train_fold_idx])
        train_subset.dataset.set_training_mode(True)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        val_subset = Subset(dataset, train_idx[val_fold_idx])
        val_subset.dataset.set_training_mode(False)
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        cv_loaders.append((train_loader, val_loader))
    
    return cv_loaders, test_loader


# Model
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

class CreateModel(nn.Module):
    def __init__(self, input_channels=5, structural_features=18,
                 denseblocks=4, layers=3, filters=96, growth_rate=32, 
                 dropout_rate=0.4):
        super().__init__()
        
        # seq特征分支
        self.initial_conv = nn.Conv1d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.denseblocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i in range(denseblocks - 1):
            self.denseblocks.append(DenseBlock(filters, layers, growth_rate, dropout_rate))
            filters += layers * growth_rate
            self.transitions.append(Transition(filters, filters // 2, dropout_rate))
            filters = filters // 2
        
        self.denseblocks.append(DenseBlock(filters, layers, growth_rate, dropout_rate))
        filters += layers * growth_rate
        
        self.cbam = CBAM(filters)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # structural特征分支
        self.structural_transformer = StructuralTransformer(
            input_dim=structural_features,
            embed_dim=64,
            num_heads=8,
            num_layers=2,
            dropout=dropout_rate)
        
        # 1.GFN fusion
        self.gate = nn.Sequential(
            nn.Linear(filters + 32, 256),
            nn.GELU(),
            nn.Linear(256, filters),
            nn.Sigmoid())
        

        # 2.可学习的加权融合
        self.structural_proj = nn.Linear(32, filters)  # 将结构特征投影到与序列特征相同维度
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 添加更多的Dropout层
        self.final_dropout = nn.Dropout(0.1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(filters + 32, 128),
            # nn.Dropout(0.1),
            nn.Linear(128, 2))
        
        self.classifier1 = nn.Sequential(
            nn.Linear(filters, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 2))
            
    
    def forward(self, x_sequence, x_structural):
        x_sequence = self.initial_conv(x_sequence)
        x_sequence = self.bn(x_sequence)
        x_sequence = self.relu(x_sequence)

        for block, transition in zip(self.denseblocks[:-1], self.transitions):
            x_sequence = block(x_sequence)
            x_sequence = transition(x_sequence)

        x_sequence = self.denseblocks[-1](x_sequence)
        x_sequence = self.cbam(x_sequence)
        x_sequence = self.avg_pool(x_sequence).squeeze(-1)

        structural_features = self.structural_transformer(x_structural)
        
        # # 1
        # combined = torch.cat([x_sequence, structural_features], dim=1)
        # gate = self.gate(combined)
        # gated_sequence = x_sequence * gate
        # final_feature = torch.cat([gated_sequence, structural_features], dim=1)

        # 2
        final_feature = torch.cat([x_sequence, structural_features], dim=1)

        # # 3
        # structural_features = self.structural_proj(structural_features)  # 投影到 [batch, filters]
        # final_feature = self.alpha * x_sequence + (1 - self.alpha) * structural_features
        # final_feature = self.final_dropout(final_feature)
        # return self.classifier1(final_feature)

        final_feature = self.final_dropout(final_feature)
        return self.classifier(final_feature)

def train_model_for_fold(model, train_loader, val_loader, criterion, optimizer, device,
                         num_epochs=50, patience=5, min_delta=0.001, fold_idx=0, output_dir='models'):
    """训练单个折的模型"""
    best_acc = 0.0
    best_metrics = {}
    no_improve_epochs = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_sn': [],
        'val_sp': [],
        'val_acc': [],
        'val_f1': [],
        'val_mcc': [],
        'val_auroc': []
    }
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'fold{fold_idx+1}.pth')
    
    scaler = GradScaler()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for ohnd, structural, labels in train_loader:
            ohnd = ohnd.to(device)
            structural = structural.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(ohnd, structural)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        val_sn, val_sp, val_acc, val_f1, val_mcc, val_auroc = evaluate_model(model, val_loader, device, verbose=False)
        history['val_sn'].append(val_sn)
        history['val_sp'].append(val_sp)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_mcc'].append(val_mcc)
        history['val_auroc'].append(val_auroc)
        
        print(f'Fold {fold_idx} | Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
              f'Val Acc: {val_acc:.4f} (Best: {best_acc:.4f}) | '
              f'SN: {val_sn:.4f} | SP: {val_sp:.4f} | '
              f'F1: {val_f1:.4f} | MCC: {val_mcc:.4f} | '
              f'AUROC: {val_auroc:.4f} | '
              f'Patience: {no_improve_epochs}/{patience}')
        
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model for fold {fold_idx} with Acc: {best_acc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1} for fold {fold_idx}')
                break
    
    model.load_state_dict(torch.load(model_path))
    
    print(f"\nFinal Validation Results for Fold {fold_idx}:")
    sn, sp, acc, f1, mcc, auroc = evaluate_model(model, val_loader, device)
    
    final_metrics = {
        'sn': sn,
        'sp': sp,
        'acc': acc,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc
    }
    training_time = time.time() - start_time
    print(f"Training completed for fold {fold_idx} in {training_time:.2f} seconds")
    
    return history, final_metrics

def evaluate_model(model, data_loader, device, verbose=True):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for ohnd, structural, labels in data_loader:
            ohnd = ohnd.to(device)
            structural = structural.to(device)
            labels = labels.to(device)
            
            outputs = model(ohnd, structural)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
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
        print(f'Metrics - SN: {sn:.4f} | SP: {sp:.4f} | ACC: {acc:.4f} | '
              f'F1: {f1:.4f} | MCC: {mcc:.4f} | AUROC: {auroc:.4f}')
        print(f'Confusion Matrix:\n{cm}')
    
    return sn, sp, acc, f1, mcc, auroc

# main
if __name__ == '__main__':

    config = {
        'pos_ohnd': 'seq_fold/data/pos_encoding_OH_ND.csv',
        'pos_struct': 'seq_fold/data/pos_fold.csv',
        'neg_ohnd': 'seq_fold/data/neg_encoding_OH_ND.csv',
        'neg_struct': 'seq_fold/data/neg_fold.csv',
        'batch_size': 128,
        'lr': 0.001,
        'num_epochs': 30,
        'patience': 15,
        'n_splits': 5,  # 5折交叉验证
        'output_dir': 'seq_fold/models/combined',
        'test_size': 0.2
    }
    
    dataset = RNADataset(
        config['pos_ohnd'],
        config['pos_struct'],
        config['neg_ohnd'],
        config['neg_struct']
    )
    
    cv_loaders, test_loader = create_cv_loaders(
        dataset,
        batch_size=config['batch_size'],
        n_splits=config['n_splits'],
        test_size=config['test_size']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(dataset)*(1-config['test_size']):.0f}, Test samples: {len(dataset)*config['test_size']:.0f}")
    
    fold_val_results = []
    fold_test_results = []
    all_histories = []
    
    for fold_idx, (train_loader, val_loader) in enumerate(cv_loaders):
        print(f"\n{'='*40}")
        print(f"Starting Fold {fold_idx + 1}/{config['n_splits']}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print('='*40)
        
        model = CreateModel(structural_features=dataset.structural_features.shape[1]).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        history, val_metrics = train_model_for_fold(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=config['num_epochs'],
            patience=config['patience'],
            fold_idx=fold_idx,
            output_dir=config['output_dir']
        )
        
        fold_val_results.append(val_metrics)  
        all_histories.append(history)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    print("\n\n=== Cross-Validation Results ===")
    metrics_names = ['sn', 'sp', 'acc', 'f1', 'mcc', 'auroc']
    
    print("\nValidation Set Performance:")
    val_summary = {metric: [] for metric in metrics_names}
    for i, metrics in enumerate(fold_val_results):
        print(f"\nFold {i}:")
        for metric in metrics_names:
            val_summary[metric].append(metrics[metric])
            print(f"{metric.upper()}: {metrics[metric]:.4f}")
    
    print("\nAverage Validation Metrics:")
    for metric in metrics_names:
        vals = val_summary[metric]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
