import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
# --- MODIFIED: Reverted back to SAGEConv ---
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
PROCESSED_DIR = 'processed_subgraphs'
# --- MODIFIED: Train for more epochs ---
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
HIDDEN_CHANNELS = 256

# --- 1. Custom Dataset Class ---
class PhishingSubgraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PhishingSubgraphDataset, self).__init__(root, transform, pre_transform)
        self.node_files = [f for f in os.listdir(self.root) if f.endswith('_nodes.csv')]
        self.scaler = MinMaxScaler()

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
            features_to_scale = nodes_df[feature_cols].values
            scaled_features = self.scaler.fit_transform(features_to_scale)
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

# --- 2. GNN Model Architecture ---
# --- MODIFIED: Reverted to GraphSAGE model ---
class GraphSAGE_4Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE_4Layer, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        self.classifier = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, root_node_idx = data.x, data.edge_index, data.root_node_idx

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))

        root_embedding = x[root_node_idx]
        output = self.classifier(root_embedding)
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

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return accuracy, precision, recall, f1

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = PhishingSubgraphDataset(root=PROCESSED_DIR)
    
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

        # Implement oversampling for the training set
        train_labels = [labels[i] for i in train_indices]
        class_counts = np.bincount(train_labels)
        if len(class_counts) > 1:
            class_weights = 1. / class_counts
            sample_weights = np.array([class_weights[label] for label in train_labels])
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- MODIFIED: Reverted to GraphSAGE model ---
        model = GraphSAGE_4Layer(in_channels=9, hidden_channels=HIDDEN_CHANNELS, out_channels=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        criterion = torch.nn.CrossEntropyLoss()

        best_val_f1 = 0
        best_model_state = None

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, train_loader, optimizer, criterion, device)
            val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, device)
            
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                torch.save(best_model_state, 'best_model.pth')
                print(f'  -> New best model saved with F1 score: {best_val_f1:.4f}')

        print("\n--- Final Test Evaluation ---")
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_loader, device)
            
            print(f'Test Accuracy: {test_acc:.4f}')
            print(f'Test Precision: {test_prec:.4f}')
            print(f'Test Recall: {test_recall:.4f}')
            print(f'Test F1 Score: {test_f1:.4f}')
        else:
            print("No best model was saved during training.")
