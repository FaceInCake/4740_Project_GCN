import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from skimage.segmentation import slic
from skimage.future import graph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pycocotools.coco import COCO
import json
import torch_geometric


# Path to the data
data_dir = './Data/'
annotation_file_training = data_dir + 'stuff_train2017.json'
annotation_file_val = data_dir + 'stuff_val2017.json'

image_dir_training = data_dir + 'train2017/'
image_dir_val = data_dir + 'val2017/'

segmentation_dir = data_dir + 'segmentations/'
val_segmentation_dir = data_dir + 'val_segmentations/'


def load_graphs_from_json(directory):
    dataset = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            path = os.path.join(directory, filename)
            with open(path, 'r') as f:
                graph_data = json.load(f)
                nx_graph = nx.node_link_graph(graph_data)
                x = torch.tensor([nx_graph.nodes[node]['features'] for node in nx_graph.nodes], dtype=torch.float)
                y = torch.tensor([max(nx_graph.nodes[node]['label']) for node in nx_graph.nodes], dtype=torch.long)  # Assuming one-hot encoding
                edge_index_list = [[src, dest] for src, dest in nx_graph.edges() if src < len(nx_graph.nodes) and dest < len(nx_graph.nodes)]
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                if edge_index.size(1) == 0:  # Skip if no edges
                    continue
                dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset



# GCN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Setup Training and Validation Procedures
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y, reduction='mean')  # `data.y` should match the output shape
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print("Output shape:", out.shape)
        # print("Labels shape:", data.y.shape)

    return total_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0  # Total number of nodes processed
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)  # Update total count

    if total == 0:
        return 0  # Avoid division by zero
    accuracy = 100 * correct / total  # Calculate as percentage
    print(f'Validating: Total Nodes={total}, Correct Predictions={correct}, Accuracy={accuracy}%')
    return accuracy



coco = COCO(annotation_file_val)
# Load the categories
categories = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in categories]
print('COCO categories: \n{}\n'.format(' '.join(json.dumps(categories))))

# Load datasets
train_dataset = load_graphs_from_json(segmentation_dir)
print('Done processing training images')
val_dataset = load_graphs_from_json(val_segmentation_dir)
print('Done processing val images')

train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
print('Done Training and Validation Data Loaders')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=3, num_classes=len(categories)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('Model and Optimizer Initialized')

# Run Training and Validation
# for epoch in range(3):
#     train_loss = train(model, train_loader, optimizer)
#     print(f'Epoch: {epoch+1}, Loss: {train_loss}')
#     val_acc = validate(model, val_loader)
#     print(f'Epoch: {epoch}, Loss: {train_loss}, Validation Accuracy: {val_acc}')

for epoch in range(3):
    train_loss = train(model, train_loader, optimizer)
    print(f'Epoch: {epoch+1}, Loss: {train_loss}')
    val_acc = validate(model, val_loader)
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
