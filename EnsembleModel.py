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
disease_metabolite_df = pd.read_csv('disease_metabolite_edgelist.csv')
disease_microbe_df = pd.read_csv('disease_microbe_edgelist.csv')
metabolite_gene_df = pd.read_csv('metabolite_gene_edgelist.csv')

# Check column names and add 'RelationType' if missing
for df in [disease_metabolite_df, disease_microbe_df, metabolite_gene_df]:
    if 'RelationType' not in df.columns:
        df['RelationType'] = df.apply(lambda row: f"{row['Source']}_{row['Target']}", axis=1)

# Combine all edge lists into one DataFrame
combined_edgelist_df = pd.concat([disease_metabolite_df, disease_microbe_df, metabolite_gene_df], ignore_index=True)

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

class SimplEDecoder(nn.Module):
    def __init__(self, out_dim, num_rels):
        super(SimplEDecoder, self).__init__()
        self.rel_embeddings = nn.Embedding(num_rels, out_dim).to(device)  
        self.rel_inv_embeddings = nn.Embedding(num_rels, out_dim).to(device)  

    def forward(self, src_embeds, dst_embeds, rel_types):
        rel_embeds = self.rel_embeddings(rel_types)  
        rel_inv_embeds = self.rel_inv_embeddings(rel_types) 
        score_1 = torch.sum(src_embeds * rel_embeds * dst_embeds, dim=1)
        score_2 = torch.sum(dst_embeds * rel_inv_embeds * src_embeds, dim=1)
        scores = torch.sigmoid((score_1 + score_2) / 2)
        return scores

class HolEDecoder(nn.Module):
    def __init__(self, out_dim, num_rels):
        super(HolEDecoder, self).__init__()
        self.rel_embeddings = nn.Embedding(num_rels, out_dim)

    def forward(self, src_embeds, dst_embeds, rel_types):
        rel_embeds = self.rel_embeddings(rel_types)
        combined_embeds = src_embeds * dst_embeds  
        real_score = torch.sum(combined_embeds * rel_embeds, dim=1)
        scores = torch.sigmoid(real_score)
        return scores

def create_target_links(G, adj_matrix):
    src, dst = np.where(adj_matrix.cpu().numpy())
    positive_links = torch.tensor(np.stack([src, dst], axis=1), device=device)
    labels = torch.ones(positive_links.size(0), device=device)

    num_neg_samples = len(positive_links)
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    negative_links = torch.stack([neg_src, neg_dst], dim=1).to(device)
    neg_labels = torch.zeros(negative_links.size(0), device=device)

    target_links = torch.cat([positive_links, negative_links], dim=0)
    target_labels = torch.cat([labels, neg_labels], dim=0)

    target_links_with_labels = torch.cat([target_links, target_labels.unsqueeze(1)], dim=1)
    return target_links_with_labels

def evaluate(links, decoders, node_embeddings, rel_types=None):
    with torch.no_grad():
        src_nodes = links[:, 0].long()
        dst_nodes = links[:, 1].long()
        labels = links[:, 2].float()

        src_embeds = node_embeddings[src_nodes]
        dst_embeds = node_embeddings[dst_nodes]

        scores = torch.zeros_like(labels)
        for decoder in decoders:
            if isinstance(decoder, (SimplEDecoder, HolEDecoder)):
                rels = rel_types[links[:, 0].long()]
                scores += decoder(src_embeds, dst_embeds, rels)
            else:
                scores += decoder(src_embeds, dst_embeds)

        scores /= len(decoders)
        preds = (scores > 0.5).float()

        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        roc_auc = roc_auc_score(labels.cpu(), scores.cpu())

    return precision, recall, f1, accuracy, roc_auc


# Initialize the encoder and decoders
encoder = GCNEncoder(num_nodes=num_nodes, h_dim=h_dim, out_dim=out_dim).to(device)
distmult_decoder = DistMultDecoder(out_dim=out_dim).to(device)
simple_decoder = SimplEDecoder(out_dim=out_dim, num_rels=num_rels).to(device)
hole_decoder = HolEDecoder(out_dim=out_dim, num_rels=num_rels).to(device)
decoders = [distmult_decoder, simple_decoder, hole_decoder]

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

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + [param for decoder in decoders for param in decoder.parameters()], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.BCELoss()
    num_epochs = 200
    early_stopping_patience = 20

    best_val_accuracy = 0
    patience_counter = 0

    train_links = train_links[torch.randperm(train_links.size(0))]

    split_idx = int(0.8 * len(train_links))
    val_links = train_links[split_idx:]
    train_links = train_links[:split_idx]

    for epoch in range(num_epochs):
        encoder.train()
        for decoder in decoders:
            decoder.train()

        optimizer.zero_grad()

        node_embeddings = encoder(adj_matrix)
        src_nodes = train_links[:, 0].long()
        dst_nodes = train_links[:, 1].long()
        labels = train_links[:, 2].float()

        src_embeds = node_embeddings[src_nodes]
        dst_embeds = node_embeddings[dst_nodes]

        total_loss = 0
        for decoder in decoders:
            if isinstance(decoder, (SimplEDecoder, HolEDecoder)):
                rels = edge_type_tensor[src_nodes]
                outputs = decoder(src_embeds, dst_embeds, rels)
            else:
                outputs = decoder(src_embeds, dst_embeds)

            loss = criterion(outputs, labels)
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        encoder.eval()
        for decoder in decoders:
            decoder.eval()

        with torch.no_grad():
            node_embeddings = encoder(adj_matrix)

            precision, recall, f1, accuracy, roc_auc = evaluate(test_links, decoders, node_embeddings, edge_type_tensor)
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

            scheduler.step(1 - accuracy)

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation accuracy.")
                break

    node_embeddings = encoder(adj_matrix)
    test_src_nodes = test_links[:, 0].long()
    test_dst_nodes = test_links[:, 1].long()
    test_labels = test_links[:, 2].float()

    test_src_embeds = node_embeddings[test_src_nodes]
    test_dst_embeds = node_embeddings[test_dst_nodes]
    rel_types = edge_type_tensor[test_src_nodes]

    test_scores = torch.zeros_like(test_labels)
    for decoder in decoders:
        if isinstance(decoder, (SimplEDecoder, HolEDecoder)):
            test_scores += decoder(test_src_embeds, test_dst_embeds, rel_types)
        else:
            test_scores += decoder(test_src_embeds, test_dst_embeds)

    test_scores /= len(decoders)
    fpr, tpr, _ = roc_curve(test_labels.cpu().detach().numpy(), test_scores.cpu().detach().numpy())
    fpr_list.append(fpr)
    tpr_list.append(interp(mean_fpr, fpr, tpr))

    precision, recall, f1, accuracy, roc_auc = evaluate(test_links, decoders, node_embeddings, edge_type_tensor)
    fold_results.append((precision, recall, f1, accuracy, roc_auc))
    print(f"Test Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

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

def save_results(results, filename='fold_results.csv'):
    df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC AUC'])
    df.to_csv(filename, index=False)

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

save_results(fold_results_dict)

plot_final_roc_curve(fpr_list, tpr_list, mean_fpr)

avg_results = np.mean(fold_results, axis=0)

# Final summary
print("\nSummary of 10-Fold Cross-Validation Results:")
for fold, result in enumerate(fold_results):
    precision, recall, f1, accuracy, roc_auc = result
    print(f"Fold {fold + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

print(f"\nAverage Results: Precision: {avg_results[0]:.4f}, Recall: {avg_results[1]:.4f}, F1-Score: {avg_results[2]:.4f}, Accuracy: {avg_results[3]:.4f}, ROC AUC: {avg_results[4]:.4f}")

predicted_labels = test_scores.cpu().detach().numpy()
true_labels = test_labels.cpu().numpy()

# Use the test set's source and destination nodes
src_nodes = test_links[:, 0].cpu().numpy()
dst_nodes = test_links[:, 1].cpu().numpy()

# Convert the predicted and true node indices back to their original labels
predicted_src_original = le.inverse_transform(src_nodes.astype(int))
predicted_dst_original = le.inverse_transform(dst_nodes.astype(int))

# Identify the top predicted relationships
top_indices = np.argsort(predicted_labels)[::-1][:]

# Create a list to store the top predicted relationships
top_predicted_relationships = []

for idx in top_indices:
    relationship = {
        'Source': predicted_src_original[idx],
        'Target': predicted_dst_original[idx],
        'Score': predicted_labels[idx]
    }
    top_predicted_relationships.append(relationship)
    print(f"{predicted_src_original[idx]} -> {predicted_dst_original[idx]} with score {predicted_labels[idx]}")

# Convert the list to a DataFrame
top_predicted_df = pd.DataFrame(top_predicted_relationships)

# Save the top predicted relationships to a CSV file
top_predicted_df.to_csv('top_predicted_relationships.csv', index=False)
print("Top predicted relationships have been saved to 'top_predicted_relationships.csv'")


existing_links = set(zip(combined_edgelist_df['Source'], combined_edgelist_df['Target']))

# Create a set of predicted links
predicted_links = set(zip(predicted_src_original, predicted_dst_original))

# Find the new links by subtracting the existing links from the predicted ones
new_links = predicted_links - existing_links

# Convert the new links to a list of dictionaries with the corresponding scores
new_predicted_relationships = [
    {'Source': src, 'Target': dst, 'Score': predicted_labels[idx]}
    for idx, (src, dst) in enumerate(zip(predicted_src_original, predicted_dst_original))
    if (src, dst) in new_links
]

# Convert the new predicted relationships to a DataFrame
new_predicted_df = pd.DataFrame(new_predicted_relationships)

# Save the new predicted relationships to a CSV file
new_predicted_df.to_csv('new_predicted_relationships.csv', index=False)
print("New predicted relationships have been saved to 'new_predicted_relationships.csv'")
