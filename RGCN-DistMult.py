import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load edge lists that obtained after running creating_edgelists.ipynb
disease_metabolic_df = pd.read_csv('disease_metabolite_edgelist.csv')
disease_microbe_df = pd.read_csv('disease_microbe_edgelist.csv')
metabolite_gene_df = pd.read_csv('metabolite_gene_edgelist.csv')

# Check column names and add 'RelationType' if missing
for df in [disease_metabolic_df, disease_microbe_df, metabolite_gene_df]:
    if 'RelationType' not in df.columns:
        df['RelationType'] = df.apply(lambda row: f"{row['Source']}_{row['Target']}", axis=1)

# Combine all edge lists into one DataFrame
combined_edgelist_df = pd.concat([disease_metabolic_df, disease_microbe_df, metabolite_gene_df], ignore_index=True)

# Create a directed graph from the combined edge list
G = nx.from_pandas_edgelist(combined_edgelist_df, source='Source', target='Target', create_using=nx.DiGraph)

# Encode the nodes (both source and target) into integers
all_nodes = list(G.nodes())
le = LabelEncoder()
le.fit(all_nodes)
encoded_nodes = le.transform(all_nodes)

# Create a mapping from node names to their encoded labels
node_mapping = {node: encoded for node, encoded in zip(all_nodes, encoded_nodes)}

# Reindex the nodes in the graph to the encoded labels
G = nx.relabel_nodes(G, node_mapping)

# Encode edge types using LabelEncoder
edge_types = combined_edgelist_df['RelationType'].values
label_encoder = LabelEncoder()
encoded_edge_types = label_encoder.fit_transform(edge_types)

# Create adjacency matrix and edge type tensor
num_nodes = len(G.nodes())
adj_matrix = nx.to_numpy_array(G, nodelist=range(num_nodes))
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
edge_type_tensor = torch.tensor(encoded_edge_types, dtype=torch.long, device=device)

# Number of nodes and relations
num_rels = len(set(encoded_edge_types))

h_dim = 128
out_dim = 64

class GCNEncoder(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim).to(device)
        self.fc1 = nn.Linear(h_dim, h_dim).to(device)
        self.fc2 = nn.Linear(h_dim, out_dim).to(device)
        self.fc3 = nn.Linear(out_dim, out_dim).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.bn1 = nn.BatchNorm1d(h_dim).to(device)
        self.bn2 = nn.BatchNorm1d(out_dim).to(device)

    def forward(self, adj_matrix):
        x = self.embedding.weight
        x = torch.matmul(adj_matrix, x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.matmul(adj_matrix, x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class DistMultDecoder(nn.Module):
    def __init__(self, out_dim):
        super(DistMultDecoder, self).__init__()
        self.rel_embeddings = nn.Parameter(torch.randn(out_dim, device=device))

    def forward(self, src_embeds, dst_embeds):
        scores = torch.sigmoid(torch.sum(src_embeds * self.rel_embeddings * dst_embeds, dim=1))
        return scores

def create_target_links(G, adj_matrix):
    src, dst = np.where(adj_matrix.cpu().numpy())
    positive_links = torch.tensor(np.stack([src, dst], axis=1), device=device)
    labels = torch.ones(positive_links.size(0), device=device)

    # negative sampling
    num_neg_samples = len(positive_links)
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    negative_links = torch.stack([neg_src, neg_dst], dim=1).to(device)
    neg_labels = torch.zeros(negative_links.size(0), device=device)

    target_links = torch.cat([positive_links, negative_links], dim=0)
    target_labels = torch.cat([labels, neg_labels], dim=0)

    target_links_with_labels = torch.cat([target_links, target_labels.unsqueeze(1)], dim=1)
    return target_links_with_labels

# Evaluation function
def evaluate(links):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src_nodes = links[:, 0].long()
        dst_nodes = links[:, 1].long()
        labels = links[:, 2].float()

        src_embeds = node_embeddings[src_nodes]
        dst_embeds = node_embeddings[dst_nodes]
        scores = decoder(src_embeds, dst_embeds)

        preds = (scores > 0.5).float()

        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        roc_auc = roc_auc_score(labels.cpu(), scores.cpu())

    return precision, recall, f1, accuracy, roc_auc, scores, labels

# Initialize the encoder and decoder
encoder = GCNEncoder(num_nodes=num_nodes, h_dim=h_dim, out_dim=out_dim).to(device)
decoder = DistMultDecoder(out_dim=out_dim).to(device)

# Generate the target links with labels
target_links = create_target_links(G, adj_matrix)

# 10-Fold Cross-Validation Setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

# Initialize lists to store FPR, TPR, and thresholds
fpr_list, tpr_list = [], []
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_idx, test_idx) in enumerate(kf.split(target_links)):
    print(f"Fold {fold+1}/{kf.get_n_splits()}")

    train_links, test_links = target_links[train_idx], target_links[test_idx]

    # Training parameters
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.BCELoss()
    num_epochs = 200
    early_stopping_patience = 20

    best_val_accuracy = 0
    patience_counter = 0

    # Shuffle train links
    train_links = train_links[torch.randperm(train_links.size(0))]

    # Split train links into training and validation sets
    split_idx = int(0.8 * len(train_links))
    val_links = train_links[split_idx:]
    train_links = train_links[:split_idx]

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        node_embeddings = encoder(adj_matrix)

        src_nodes = train_links[:, 0].long()
        dst_nodes = train_links[:, 1].long()
        labels = train_links[:, 2].float()

        src_embeds = node_embeddings[src_nodes]
        dst_embeds = node_embeddings[dst_nodes]
        scores = decoder(src_embeds, dst_embeds)

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        # Validation
        val_precision, val_recall, val_f1, val_accuracy, _, _, _ = evaluate(val_links)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        scheduler.step(loss)

        # Check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Test evaluation
    test_precision, test_recall, test_f1, test_accuracy, test_roc_auc, test_scores, test_labels = evaluate(test_links)
    fold_results.append((test_precision, test_recall, test_f1, test_accuracy, test_roc_auc))
    print(f"Test Results - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, ROC AUC: {test_roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(test_labels.cpu(), test_scores.cpu())
    fpr_list.append(fpr)
    tpr_list.append(interp(mean_fpr, fpr, tpr))

# Function to plot final averaged ROC Curve
def plot_final_roc_curve(fpr_list, tpr_list, mean_fpr):
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the last point of the ROC curve is (1, 1)
    mean_auc = roc_auc_score(np.hstack([np.ones(len(mean_fpr) // 2), np.zeros(len(mean_fpr) // 2)]), mean_tpr)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.4f})')
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tpr_list, axis=0), 0), np.minimum(mean_tpr + np.std(tpr_list, axis=0), 1), color='blue', alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final Averaged ROC Curve')
    plt.show()

# Function to save results to a CSV file
def save_results(results, filename='DistMult_fold_results.csv'):
    df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC AUC'])
    df.to_csv(filename, index=False)

# Store the results
fold_results_dict = []
for fold in range(len(fold_results)):
    precision, recall, f1, accuracy, roc_auc = fold_results[fold]
    fold_results_dict.append({
        'Fold': fold + 1,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'ROC AUC': roc_auc
    })

# Save fold results
save_results(fold_results_dict)

# Plot final averaged ROC curve
plot_final_roc_curve(fpr_list, tpr_list, mean_fpr)

avg_results = np.mean(fold_results, axis=0)

# Final summary
print("\nSummary of 10-Fold Cross-Validation Results:")
for fold, result in enumerate(fold_results):
    precision, recall, f1, accuracy, roc_auc = result
    print(f"Fold {fold + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

print(f"\nAverage Results: Precision: {avg_results[0]:.4f}, Recall: {avg_results[1]:.4f}, F1-Score: {avg_results[2]:.4f}, Accuracy: {avg_results[3]:.4f}, ROC AUC: {avg_results[4]:.4f}")
