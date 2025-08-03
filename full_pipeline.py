import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# --- Configuration for the Best Performing Model ---
PROCESSED_DIR = 'processed_subgraphs'
NUM_EPOCHS = 70
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
HIDDEN_CHANNELS = 256

# --- 1. Custom Dataset Class ---
class PhishingSubgraphDataset(Dataset):
    def __init__(self, root, scaler, transform=None, pre_transform=None):
        super(PhishingSubgraphDataset, self).__init__(root, transform, pre_transform)
        self.node_files = [f for f in os.listdir(self.root) if f.endswith('_nodes.csv')]
        self.scaler = scaler

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.node_files

    def len(self):
        return len(self.node_files)

    def get(self, idx):
        node_filename = self.node_files[idx]
        base_name = node_filename.replace('_nodes.csv', '')
        edge_filename = base_name + '_edges.csv'
        label_str = base_name.split('_')[-1]
        
        nodes_df = pd.read_csv(os.path.join(self.root, node_filename))
        edges_df = pd.read_csv(os.path.join(self.root, edge_filename))

        address_to_int = {address: i for i, address in enumerate(nodes_df['address'])}
        
        feature_cols = [
            'eth_sent', 'eth_received', 'num_sent_txs', 'num_received_txs', 
            'total_txs', 'account_age_days', 'avg_time_diff_hours', 
            'avg_tx_value', 'std_dev_tx_value'
        ]
        for col in feature_cols:
            if col not in nodes_df.columns:
                nodes_df[col] = 0
        nodes_df = nodes_df.fillna(0)

        if not nodes_df.empty:
            skewed_features = ['eth_sent', 'eth_received', 'avg_tx_value', 'std_dev_tx_value']
            for col in skewed_features:
                nodes_df[col] = np.log1p(nodes_df[col])

            features_to_scale = nodes_df[feature_cols].values
            scaled_features = self.scaler.transform(features_to_scale)
            x = torch.tensor(scaled_features, dtype=torch.float)
        else:
            x = torch.empty((0, len(feature_cols)), dtype=torch.float)

        src, dst = [], []
        for _, edge in edges_df.iterrows():
            from_addr = edge['from_address']
            to_addr = edge['to_address']
            if from_addr in address_to_int and to_addr in address_to_int:
                src.append(address_to_int[from_addr])
                dst.append(address_to_int[to_addr])
        
        if not src:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        y = torch.tensor([1 if label_str == 'phishing' else 0], dtype=torch.long)
        
        root_address = base_name.split('_')[0]
        root_node_idx = address_to_int.get(root_address, 0)

        data = Data(x=x, edge_index=edge_index, y=y, root_node_idx=root_node_idx)
        return data

# --- NEW: Focal Loss Class ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

# --- 2. GNN Model Architecture ---
class GraphSAGE_4Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE_4Layer, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        self.classifier = Linear(hidden_channels * 2, out_channels)
        self.dropout = Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, root_node_idx, batch = data.x, data.edge_index, data.root_node_idx, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))

        root_embedding = x[root_node_idx]
        graph_embedding = global_mean_pool(x, batch)
        combined_embedding = torch.cat([root_embedding, graph_embedding], dim=1)
        
        output = self.classifier(combined_embedding)
        return output

# --- 3. Training and Evaluation Logic ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# --- MODIFIED: Evaluate function now returns probabilities for threshold tuning ---
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            # Get probabilities for the 'phishing' class
            probs = F.softmax(out, dim=1)[:, 1] 
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    return np.array(all_probs), np.array(all_labels)

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Pre-calculating global feature statistics for normalization...")
    all_node_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith('_nodes.csv')]
    
    all_nodes_df = pd.concat([pd.read_csv(f) for f in all_node_files])
    
    feature_cols = [
        'eth_sent', 'eth_received', 'num_sent_txs', 'num_received_txs', 
        'total_txs', 'account_age_days', 'avg_time_diff_hours', 
        'avg_tx_value', 'std_dev_tx_value'
    ]
    for col in feature_cols:
        if col not in all_nodes_df.columns:
            all_nodes_df[col] = 0
    all_nodes_df = all_nodes_df.fillna(0)
    
    skewed_features = ['eth_sent', 'eth_received', 'avg_tx_value', 'std_dev_tx_value']
    for col in skewed_features:
        all_nodes_df[col] = np.log1p(all_nodes_df[col])

    global_scaler = MinMaxScaler()
    global_scaler.fit(all_nodes_df[feature_cols].values)
    print("Global scaler has been fitted on log-transformed data.")

    dataset = PhishingSubgraphDataset(root=PROCESSED_DIR, scaler=global_scaler)
    
    if len(dataset) == 0:
        print("Dataset is empty. Please check your 'processed_subgraphs' directory.")
    else:
        indices = list(range(len(dataset)))
        labels = [dataset.get(i).y.item() for i in indices]
        
        train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
        
        test_val_labels = [labels[i] for i in test_indices]
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42, stratify=test_val_labels)

        train_dataset = dataset[train_indices]
        val_dataset = dataset[val_indices]
        test_dataset = dataset[test_indices]
        
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = GraphSAGE_4Layer(in_channels=9, hidden_channels=HIDDEN_CHANNELS, out_channels=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        criterion = FocalLoss().to(device)
        
        best_val_f1 = 0
        best_model_state = None

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, train_loader, optimizer, criterion, device)
            val_probs, val_labels = evaluate(model, val_loader, device)
            # Use default 0.5 threshold for validation during training
            val_preds = (val_probs > 0.5).astype(int)
            val_f1 = f1_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                torch.save(best_model_state, 'best_model.pth')
                print(f'  -> New best model saved with F1 score: {best_val_f1:.4f}')

        print("\n--- Final Test Evaluation ---")
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            
            # --- NEW: Find the best threshold on the validation set ---
            print("\nFinding optimal threshold on validation set...")
            val_probs, val_labels = evaluate(model, val_loader, device)
            best_threshold = 0
            best_f1 = 0
            for threshold in np.arange(0.1, 1.0, 0.05):
                preds = (val_probs > threshold).astype(int)
                f1 = f1_score(val_labels, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            print(f"Optimal threshold found: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

            # --- NEW: Evaluate on the test set using the optimal threshold ---
            print("\nEvaluating on test set with optimal threshold...")
            test_probs, test_labels = evaluate(model, test_loader, device)
            test_preds = (test_probs > best_threshold).astype(int)
            
            test_acc = accuracy_score(test_labels, test_preds)
            test_prec = precision_score(test_labels, test_preds)
            test_recall = recall_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds)

            print(f'Test Accuracy: {test_acc * 100:.2f}%')
            print(f'Test Precision: {test_prec * 100:.2f}%')
            print(f'Test Recall: {test_recall * 100:.2f}%')
            print(f'Test F1 Score: {test_f1 * 100:.2f}%')
        else:
            print("No best model was saved during training.")
