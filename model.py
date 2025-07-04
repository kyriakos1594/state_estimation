import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, global_max_pool, GlobalAttention
from model_components import *
from config_file import *
import numpy as np
import sys
from torch_geometric.utils import to_dense_batch
#TODO To define the Transformer Encoder and Transformer Decoder layers manually without relying
#      on TransformerEncoderLayer and TransformerDecoderLayer from PyTorch, we can break down the individual
#      components of a Transformer layer, which are:
#      Multihead Self-Attention: Each token (node feature in our case) is transformed by attending to every other
#      token in the sequence (graph) via a weighted sum (attention mechanism).
#      Feedforward Neural Network (FFN): After the attention step, a position-wise fully connected feedforward network
#      is applied to each token. Normalization: After each sub-layer (self-attention and FFN), layer normalization
#      is applied to stabilize training. Residual Connections: The output of each sub-layer is added back to the input,
#      creating residual connections.

print_flag = True
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=float('inf'))

#TODO TI NN FOr all cases
class TI_SimpleNNEdges(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, branch_num=None, branch_feature_num=None):
        super(TI_SimpleNNEdges, self).__init__()
        self.branch_num = branch_num
        self.branch_feature_num = branch_feature_num
        if self.branch_feature_num is not None:
            self.input_dim = num_nodes * num_features + branch_num * branch_feature_num  # Flatten the entire graph input
            print("Input dimension: ", self.input_dim)
        else:
            self.input_dim = num_nodes * num_features # Flatten the entire graph input
            print("Input dimension: ", self.input_dim)

        print("Input dimension for SimpleNN: ", self.input_dim, "nodes: ", num_nodes, "features: ", num_features, "branches: ", branch_num, "features (branches): ", branch_feature_num)

        # Fully connected layers
        #self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(self.input_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        #self.fc4 = nn.Linear(128, 64)
        #self.fc5 = nn.Linear(64, num_classes)
        #self.fc4 = nn.Linear(4, num_classes)

        # Dropout for regularization
        #self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten the input (batch_size, num_nodes, num_features) -> (batch_size, num_nodes * num_features)
        #print("In net:", x)
        x = x.view(x.size(0), -1)
        #print("In net view: ", x)

        # Forward pass through MLP
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        # Output layer (logits)
        #x = self.fc4(x)

        return x #F.log_softmax(x, dim=1)  # No softmax, using CrossEntropyLoss

#TODO SE NN For all cases
class SE_SimpleNNEdges(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, branch_num=None, branch_feature_num=None):
        super(SE_SimpleNNEdges, self).__init__()
        self.branch_num = branch_num
        self.branch_feature_num = branch_feature_num
        if self.branch_feature_num is not None:
            self.input_dim = num_nodes * num_features + branch_num * branch_feature_num  # Flatten the entire graph input
            print("Input dimension: ", self.input_dim)
        else:
            self.input_dim = num_nodes * num_features # Flatten the entire graph input
            print("Input dimension: ", self.input_dim)

        print("Input dimension for SimpleNN: ", self.input_dim, "nodes: ", num_nodes, "features: ", num_features, "branches: ", branch_num, "features (branches): ", branch_feature_num)

        # Fully connected layers
        #self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 95)
        self.fc3 = nn.Linear(95, output_dim)
        #self.fc4 = nn.Linear(128, 64)
        #self.fc5 = nn.Linear(64, num_classes)
        #self.fc4 = nn.Linear(4, num_classes)

        # Dropout for regularization
        #self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten the input (batch_size, num_nodes, num_features) -> (batch_size, num_nodes * num_features)
        #print("In net:", x)
        x = x.view(x.size(0), -1)
        #print("In net view: ", x)

        # Forward pass through MLP
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output layer (logits)
        #x = self.fc4(x)

        return x #F.log_softmax(x, dim=1)  # No softmax, using CrossEntropyLoss

#TODO SE NN for all cases
class SE_SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SE_SimpleNN, self).__init__()

        # Define the layers
        #self.fc1 = nn.Linear(input_dim, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(input_dim, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Pass the input through the layers
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = torch.nn.relu(self.fc4(x))
        x = torch.nn.relu(self.fc5(x))
        x = self.fc6(x)  # Output layer (no activation for regression)
        return x

#TODO TI GNN GAT - Only PMU_caseA
class TI_GATWithEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes, edge_attr_dim, gat_layers=1, GAT_dim=8, heads=4):
        super(TI_GATWithEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features) - Input
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True)  # No edge_dim

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True) for _ in range(gat_layers-1)
        ])

        self.fc1 = torch.nn.Linear(GAT_dim * heads, 2 * GAT_dim)

        # Fully connected layer for classification
        self.fc2 = torch.nn.Linear(2 * GAT_dim, num_classes)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        batch = batch.to(self.device)


        x = self.input_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        for gatconv in self.GATConv_layers:
            x = gatconv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = global_max_pool(x, batch)


        # Fully connected layer: Output the final classes
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x #F.log_softmax(x, dim=1)

class TI_GATWithEdgeAttrNodeProj(torch.nn.Module):
    def __init__(self, num_nodes, proj_nodes, num_features, output_dim, edge_attr_dim, GAT_dim=16, gat_layers=3, heads=4):
        super(TI_GATWithEdgeAttrNodeProj, self).__init__()

        self.num_nodes = num_nodes
        self.proj_nodes = proj_nodes

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True, edge_dim=edge_attr_dim)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True) for _ in range(gat_layers-1)
        ])

        # Fully connected layer for classification
        self.node_proj = nn.Linear(GAT_dim * heads, 1 * self.proj_nodes)
        self.fc_out = torch.nn.Linear(num_nodes * self.proj_nodes, output_dim)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #print("x input: ", x.shape)

        x = self.input_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        #print("x after input GATCONV shape: ", x.shape)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            #print("x after GATCONV shape: ", x.shape)

        x = self.node_proj(x)
        x = F.relu(x)
        #print("x after node projection shape: ", x.shape)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)
        #print("x after node reshape: ", x.shape)

        x = self.fc_out(x)
        #print("x output: ", x.shape)

        return x

class TI_TEGNN_WithEdges(nn.Module):
    def __init__(self, device, num_nodes, num_features, output_dim, proj_dim=4, embedding_dim=4, heads=4,
                 num_decoder_layers=1, edge_attr_dim=2, gat_layers=4, GATConv_dim=16):
        super(TI_TEGNN_WithEdges, self).__init__()

        self.num_nodes = num_nodes
        self.device = device

        # Embedding for node indices (shared across batches)
        self.node_index_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # Update feature_fc to take into account the concatenated input size
        self.feature_fc = nn.Linear(num_features + embedding_dim, 3*embedding_dim)

        # Input GATConv layer
        self.input_gat_conv = GATConv(3*embedding_dim, GATConv_dim, edge_dim=edge_attr_dim, heads=heads, concat=True)

        # GATConv stacking for graph feature extraction
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, edge_dim=edge_attr_dim, heads=heads, concat=True)
            for _ in range(gat_layers-1)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            #TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim)
            nn.MultiheadAttention(embed_dim=GATConv_dim * heads, num_heads=heads, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        # Node projection layer
        self.node_proj_FC = nn.Linear(GATConv_dim * heads, proj_dim)

        # Fully connected layer for output (e.g., classification or regression)
        self.out = nn.Linear(proj_dim * self.num_nodes, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = int(batch.max().item()) + 1

        node_indices = torch.arange(self.num_nodes, device=self.device).repeat(batch.max().item() + 1)
        #print("Shape of node indices: ", node_indices.shape)

        # Get node index embeddings
        index_embeds = self.node_index_embedding(node_indices)  # [total_nodes, embedding_dim]
        #print("Shape of index embeddings: ", index_embeds.shape)

        # Concatenate raw features with index embeddings
        x = torch.cat([x, index_embeds], dim=1)  # Shape: [total_nodes, num_features + embedding_dim]
        #print("Concatenation of input after embedding", x.shape)

        #print("Shape before feature FC: ", x.shape)
        # Embedding node features if needed - input
        x = self.feature_fc(x)
        #print("Feature FC after shape: ", x.shape)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x)
        #print("Shape x after input GATConv: ", x.shape)

        # Local graph feature extraction, using graph attention layers
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            #print("Shape x after GATConv: ", x.shape)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        #print("Shape x after decoder squeeze: ", x.shape)

        x = x.reshape(batch_size,self.num_nodes,-1)
        #print("Shape x after batch reshape: ", x.shape)

        # Global attention applied through transformer
        for decoder in self.transformer_decoder:
            x, attn_weights = decoder(query=x,
                                      key=x,
                                      value=x)
            #print("Shape x after decoder layer: ", x.shape)

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)
        #print("Shape x after decoder squeeze: ", x.shape)

        x = self.node_proj_FC(x)
        #print("Shape after node projection", x.shape)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)
        #print("Shape after batch projection", x.shape)

        # Final fully connected layer for the output
        x = self.out(x)
        #print("Shape of output: ", x.shape)

        return x

#TODO TI GNN GAT - PMU_caseB or conventional
class TI_GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_gat_layers=6, gat_dim=16, heads=4):
        super(TI_GATNoEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features) - Input
        self.input_conv = GATConv(num_features, gat_dim, heads=heads, concat=True)  # No edge_dim

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(gat_dim * heads, gat_dim, heads=heads, concat=True) for _ in range(num_gat_layers-1)
        ])

        self.fc1 = torch.nn.Linear(gat_dim * heads, 2 * gat_dim)

        # Fully connected layer for classification
        self.fc2 = torch.nn.Linear(2 * gat_dim, num_classes)

    def forward(self, data):
        # If no node features exist, initialize dummy features
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)

        # First GAT layer
        x = self.input_conv(x, edge_index=edge_index)
        x = F.relu(x)

        # Second GAT layer
        for gatconv_layer in self.GATConv_layers:
            x = gatconv_layer(x, edge_index=edge_index)
            x = F.relu(x)

        x = global_max_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class TI_GATNoEdgeAttrNodeProj(torch.nn.Module):
    def __init__(self, num_nodes, proj_nodes, num_features, output_dim, GAT_dim=16, gat_layers=3,
                 heads=4):
        super(TI_GATNoEdgeAttrNodeProj, self).__init__()

        self.num_nodes = num_nodes
        self.proj_nodes = proj_nodes

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, concat=True) for _ in
            range(gat_layers - 1)
        ])

        # Fully connected layer for classification
        self.node_proj = nn.Linear(GAT_dim * heads, 1 * self.proj_nodes)
        self.fc_out = torch.nn.Linear(num_nodes * self.proj_nodes, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print("x input: ", x.shape)

        x = self.input_conv(x, edge_index=edge_index)
        x = F.relu(x)
        #print("x after input GATCONV shape: ", x.shape)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index)
            x = F.relu(x)
            # print("x after GATCONV shape: ", x.shape)

        x = self.node_proj(x)
        x = F.relu(x)
        #print("x after node projection shape: ", x.shape)

        batch_size = int(batch.max().item()) + 1
        x = x.reshape(batch_size, -1)
        #print("x after node reshape: ", x.shape)

        x = self.fc_out(x)
        #print("x output: ", x.shape)

        return x


class TI_TEGNN_NoEdges(nn.Module):
    def __init__(self, device, num_nodes, num_features, output_dim, proj_dim=4, embedding_dim=4, heads=4,
                 num_decoder_layers=1, gat_layers=4, GATConv_dim=16):
        super(TI_TEGNN_NoEdges, self).__init__()

        self.num_nodes = num_nodes
        self.device = device

        # Embedding for node indices (shared across batches)
        self.node_index_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # Update feature_fc to take into account the concatenated input size
        self.feature_fc = nn.Linear(num_features + embedding_dim, 3*embedding_dim)

        # Input GATConv layer
        self.input_gat_conv = GATConv(3*embedding_dim, GATConv_dim, heads=heads, concat=True)

        # GATConv stacking for graph feature extraction
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, heads=heads, concat=True) for _ in range(gat_layers-1)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            #TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim)
            nn.MultiheadAttention(embed_dim=GATConv_dim * heads, num_heads=heads, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        # Node projection layer
        self.node_proj_FC = nn.Linear(GATConv_dim * heads, proj_dim)

        # Fully connected layer for output (e.g., classification or regression)
        self.out = nn.Linear(proj_dim * self.num_nodes, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = int(batch.max().item()) + 1

        node_indices = torch.arange(self.num_nodes, device=self.device).repeat(batch.max().item() + 1)
        #print("Shape of node indices: ", node_indices.shape)

        # Get node index embeddings
        index_embeds = self.node_index_embedding(node_indices)  # [total_nodes, embedding_dim]
        #print("Shape of index embeddings: ", index_embeds.shape)

        # Concatenate raw features with index embeddings
        x = torch.cat([x, index_embeds], dim=1)  # Shape: [total_nodes, num_features + embedding_dim]
        #print("Concatenation of input after embedding", x.shape)

        #print("Shape before feature FC: ", x.shape)
        # Embedding node features if needed - input
        x = self.feature_fc(x)
        #print("Feature FC after shape: ", x.shape)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index)
        x = F.leaky_relu(x)
        #print("Shape x after input GATConv: ", x.shape)

        # Local graph feature extraction, using graph attention layers
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index)
            x = F.leaky_relu(x)
            #print("Shape x after GATConv: ", x.shape)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        #print("Shape x after decoder squeeze: ", x.shape)

        x = x.reshape(batch_size,self.num_nodes,-1)
        #print("Shape x after batch reshape: ", x.shape)

        # Global attention applied through transformer
        for decoder in self.transformer_decoder:
            x, attn_weights = decoder(query=x,
                                      key=x,
                                      value=x)
            #print("Shape x after decoder layer: ", x.shape)

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)
        #print("Shape x after decoder squeeze: ", x.shape)

        x = self.node_proj_FC(x)
        #print("Shape after node projection", x.shape)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)
        #print("Shape after batch projection", x.shape)

        # Final fully connected layer for the output
        x = self.out(x)
        #print("Shape of output: ", x.shape)

        return x



#TODO TI Transformer based - PMU_caseB or conventional
class TI_TransformerNoEdges(torch.nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, GATConv_layers, GATConv_dim, embedding_dim=4, heads=4, dec_layers=1, ff_hid_dim=32):
        super(TI_TransformerNoEdges, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_nodes = num_nodes

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_features, embedding_dim)
        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Convolution Layers (Graph Attention) - Input
        self.input_gat_conv = GATConv(embedding_dim, GATConv_dim, heads=heads, concat=True)

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, heads=heads, concat=True) for _ in range(GATConv_layers-1)
        ])

        # Custom Transformer Decoder Layer
        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads//2, dim_feedforward=ff_hid_dim) for _ in range(dec_layers)
        ])

        #self.attn_pool = GlobalAttention(in_channels)

        # Fully connected layer for output (e.g., classification or regression)
        # Final output layer
        self.fc = nn.Linear(GATConv_dim * heads, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Recover shape info
        batch_size = int(batch.max().item()) + 1

        # Embedding node features if needed - input
        x = self.feature_fc(x)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index)
        x = F.leaky_relu(x)

        layers = 0
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index)
            layers += 1
            x = F.leaky_relu(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        x = x.reshape(batch_size, self.num_nodes, -1)

        for decoder in self.transformer_decoder:
            x = decoder(x, x)  # Self-attention within the graph

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.reshape(batch_size * self.num_nodes, -1)

        # Apply global pooling: aggregate node-level features into graph-level features
        x_global = global_mean_pool(x, batch)

        x = self.fc(x_global)

        return x

#TODO TI Transformer based - PMU_caseA
class TI_TransformerWithEdges(torch.nn.Module):
    def __init__(self, num_features, num_classes, gat_layers, GAT_dim, edge_attr_dim=2, embedding_dim=4, heads=4, dec_layers=1, ff_hid_dim=24):
        super(TI_TransformerWithEdges, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature Transformation - Fully Connected Layer
        self.feature_fc = nn.Linear(num_features, embedding_dim)

        # Graph Attention (without edge features) - Input
        self.input_conv = GATConv(embedding_dim, GAT_dim, heads=heads, edge_dim=edge_attr_dim,
                                  concat=True)  # No edge_dim

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True) for _ in
            range(gat_layers-1)
        ])

        # Decoder layers
        self.transformer_decoder = nn.ModuleList([
            TransfromerDecoderLayer2(d_model=GAT_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in
            range(dec_layers)
        ])

        #TODO - Functions better for SE (classification prefers max pooling - simpler problem)
        # self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(GAT_dim * heads, ff_hid_dim),
        #                                                         torch.nn.ReLU(),
        #                                                         torch.nn.Linear(ff_hid_dim, 1)))

        self.fc1 = torch.nn.Linear(GAT_dim * heads, 2 * GAT_dim)

        # Fully connected layer for classification
        self.fc2 = torch.nn.Linear(2 * GAT_dim, num_classes)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        batch = batch.to(self.device)

        x = self.feature_fc(x)

        x = self.input_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        for gatconv in self.GATConv_layers:
            x = gatconv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        # Global attention appliued through transformer
        for decoder in self.transformer_decoder:
            x = decoder(x, x)
        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)

        #x = self.attn_pool(x, batch)
        x = global_max_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x  # F.log_softmax(x, dim=1)

#TODO SE

#TODO GATConv - PMU_caseB, conventional
class SE_GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, output_dim, GAT_dim=16, gat_layers=3, heads=4):
        super(SE_GATNoEdgeAttrs, self).__init__()

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, concat=True) for _ in range(gat_layers-1)
        ])

        # Fully connected layer for classification
        self.fc1 = torch.nn.Linear(GAT_dim * heads, output_dim) #GAT_dim * heads, 10 * GAT_dim)
        #self.fc2 = torch.nn.Linear(10 * GAT_dim, output_dim)


    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        #print("Before input conv: ", x)
        x = self.input_conv(x, edge_index=edge_index)
        #print("after input conv: ", x)
        x = F.relu(x)

        layer = 0
        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index)
            #print(f"after conv {layer}: ", x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)

        return x

class SE_GATNoEdgeAttrsNodeProj(torch.nn.Module):
    def __init__(self, num_nodes, proj_nodes, num_features, output_dim, GAT_dim=16, gat_layers=3, heads=4):
        super(SE_GATNoEdgeAttrsNodeProj, self).__init__()

        self.num_nodes = num_nodes
        self.proj_nodes = proj_nodes

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, concat=True) for _ in range(gat_layers-1)
        ])

        # Fully connected layer for classification
        self.node_proj = nn.Linear(GAT_dim * heads, 1 * self.proj_nodes)
        self.fc_out = torch.nn.Linear(num_nodes * self.proj_nodes, output_dim)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_conv(x, edge_index=edge_index)
        x = F.relu(x)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index)
            x = F.relu(x)

        x = self.node_proj(x)
        x = F.relu(x)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)

        x = self.fc_out(x)


        return x

class SE_TEGNN_NoEdges(nn.Module):
    def __init__(self, device, num_nodes, num_features, output_dim, proj_dim=4, embedding_dim=4, heads=4,
                 num_decoder_layers=1, gat_layers=4, GATConv_dim=16, ff_hid_dim=64):
        super(SE_TEGNN_NoEdges, self).__init__()

        self.num_nodes = num_nodes
        self.device = device

        # Embedding for node indices (shared across batches)
        self.node_index_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # Update feature_fc to take into account the concatenated input size
        self.feature_fc = nn.Linear(num_features + embedding_dim, embedding_dim)

        # Node embedding layer (if needed, otherwise use raw features)
        #self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        #self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # Input GATConv layer
        self.input_gat_conv = GATConv(embedding_dim, GATConv_dim, heads=heads, concat=True)

        # GATConv stacking for graph feature extraction
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, heads=heads, concat=True)
            for _ in range(gat_layers-1)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            #TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim)
            nn.MultiheadAttention(embed_dim=GATConv_dim * heads, num_heads=heads, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        #self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(GATConv_dim * heads, 2 * GATConv_dim),
        #                                                         torch.nn.ReLU(),
        #                                                         torch.nn.Linear(2 * GATConv_dim, 1)))

        self.node_proj_FC = nn.Linear(GATConv_dim * heads, proj_dim)

        # Fully connected layer for output (e.g., classification or regression)
        self.out = nn.Linear(proj_dim * output_dim, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = int(batch.max().item()) + 1

        node_indices = torch.arange(self.num_nodes, device=self.device).repeat(batch.max().item() + 1)
        #print("Shape of node indices: ", node_indices.shape)

        # Get node index embeddings
        index_embeds = self.node_index_embedding(node_indices)  # [total_nodes, embedding_dim]
        #print("Shape of index embeddings: ", index_embeds.shape)

        # Concatenate raw features with index embeddings
        x = torch.cat([x, index_embeds], dim=1)  # Shape: [total_nodes, num_features + embedding_dim]
        #print("Concatenation of input after embedding", x.shape)

        #print("Shape before feature FC: ", x.shape)
        # Embedding node features if needed - input
        x = self.feature_fc(x)
        #print("Feature FC after shape: ", x.shape)

        #print("Shape after feature FC: ", x.shape)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index)
        x = F.leaky_relu(x)
        #print("Shape x after input GATConv: ", x.shape)

        # Local graph feature extraction, using graph attention layers
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index)
            x = F.leaky_relu(x)
            #print("Shape x after GATConv: ", x.shape)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        #print("Shape x after decoder squeeze: ", x.shape)

        x = x.reshape(batch_size,self.num_nodes,-1)
        #print("Shape x after batch reshape: ", x.shape)

        # Global attention applied through transformer
        for decoder in self.transformer_decoder:
            x, attn_weights = decoder(query=x,
                                      key=x,
                                      value=x)
            #print("Shape x after decoder layer: ", x.shape)
            #print(x[0, :])
            #print(x[0, :])

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)
        #print("Shape x after decoder squeeze: ", x.shape)

        x = self.node_proj_FC(x)
        #print("Shape after node projection", x.shape)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)
        #print("Shape after batch projection", x.shape)

        # Final fully connected layer for the output
        x = self.out(x)
        #print("Shape of output: ", x.shape)

        return x


#TODO GATConv - PMU_caseA
class SE_GATWithEdgeAttr(torch.nn.Module):
    def __init__(self, num_features, output_dim, edge_attr_dim, GAT_dim=16, gat_layers=3, heads=4):
        super(SE_GATWithEdgeAttr, self).__init__()

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True, edge_dim=edge_attr_dim)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True) for _ in range(gat_layers-1)
        ])

        # Fully connected layer for classification
        self.fc1 = torch.nn.Linear(GAT_dim * heads, output_dim)
        #self.fc2 = torch.nn.Linear(5 * GAT_dim, output_dim)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)

        return x

class SE_GATWithEdgeAttrNodeProj(torch.nn.Module):
    def __init__(self, num_nodes, proj_nodes, num_features, output_dim, edge_attr_dim, GAT_dim=16, gat_layers=3, heads=4):
        super(SE_GATWithEdgeAttrNodeProj, self).__init__()

        self.num_nodes = num_nodes
        self.proj_nodes = proj_nodes

        # Input Graph Attention layer
        # Here, `edge_attr_dim` is the size of the edge features
        self.input_conv = GATConv(num_features, GAT_dim, heads=heads, concat=True, edge_dim=edge_attr_dim)  # First GAT layer with edge features

        # Hidden Graph Attention Layers
        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GAT_dim * heads, GAT_dim, heads=heads, edge_dim=edge_attr_dim, concat=True) for _ in range(gat_layers-1)
        ])

        # Fully connected layer for classification
        self.node_proj = nn.Linear(GAT_dim * heads, 1 * self.proj_nodes)
        self.fc_out = torch.nn.Linear(num_nodes * self.proj_nodes, output_dim)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = self.node_proj(x)
        x = F.relu(x)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)

        x = self.fc_out(x)


        return x

#TODO TRANSFORMER-BASED GNNs SE

#TODO Only Encoder Transformer - PMU_caseB and conventional
class SE_GATTransfomerOnlyDecoderNoEdges(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4,
                 num_decoder_layers=1, gat_layers =4, GATConv_dim=16, ff_hid_dim=64):
        super(SE_GATTransfomerOnlyDecoderNoEdges, self).__init__()

        # Node embedding layer (if needed, otherwise use raw features)
        #self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.num_nodes = num_nodes
        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # Input GATConv layer
        self.input_gat_conv = GATConv(embedding_dim, GATConv_dim, heads=heads, concat=True)

        # GATConv stacking for graph feature extraction
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, heads=heads, concat=True) for _ in range(gat_layers-1)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads//2, dim_feedforward=ff_hid_dim) for _ in range(num_decoder_layers)
        ])

        #TODO - Functions better for SE (classification prefers max pooling - simpler problem)
        #self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(GATConv_dim * heads, GATConv_dim * heads // 2),
        #                                                         torch.nn.ReLU(),
        #                                                         torch.nn.Linear(GATConv_dim * heads // 2, 1)))

        self.fc = nn.Linear(GATConv_dim * heads, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Recover shape info
        batch_size = int(batch.max().item()) + 1

        # Embedding node features if needed - input
        x = self.feature_fc(x)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index)
        x = F.leaky_relu(x)

        layers=0
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index)
            layers+=1
            x = F.leaky_relu(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        x = x.reshape(batch_size, self.num_nodes, -1)

        for decoder in self.transformer_decoder:
            x = decoder(x, x)  # Self-attention within the graph

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.reshape(batch_size * self.num_nodes, -1)

        # Apply global pooling: aggregate node-level features into graph-level features
        x_global = global_mean_pool(x, batch)

        x = self.fc(x_global)
        return x

#TODO
class SE_TEGNN_WithEdges(nn.Module):
    def __init__(self, device, num_nodes, num_features, output_dim, proj_dim=4, embedding_dim=4, heads=4,
                 num_decoder_layers=1, edge_attr_dim=2, gat_layers=4, GATConv_dim=16, ff_hid_dim=64):
        super(SE_TEGNN_WithEdges, self).__init__()

        self.num_nodes = num_nodes
        self.device = device

        # Embedding for node indices (shared across batches)
        self.node_index_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # Update feature_fc to take into account the concatenated input size
        self.feature_fc = nn.Linear(num_features + embedding_dim, 2*embedding_dim)

        # Node embedding layer (if needed, otherwise use raw features)
        #self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        #self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # Input GATConv layer
        self.input_gat_conv = GATConv(2*embedding_dim, GATConv_dim, edge_dim=edge_attr_dim, heads=heads, concat=True)

        # GATConv stacking for graph feature extraction
        self.gatconv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, edge_dim=edge_attr_dim, heads=heads, concat=True)
            for _ in range(gat_layers-1)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            #TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim)
            nn.MultiheadAttention(embed_dim=GATConv_dim * heads, num_heads=heads, batch_first=True)
            for _ in range(num_decoder_layers)
        ])

        #self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(GATConv_dim * heads, 2 * GATConv_dim),
        #                                                         torch.nn.ReLU(),
        #                                                         torch.nn.Linear(2 * GATConv_dim, 1)))

        self.node_proj_FC = nn.Linear(GATConv_dim * heads, proj_dim)

        # Fully connected layer for output (e.g., classification or regression)
        self.out = nn.Linear(proj_dim * output_dim, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = int(batch.max().item()) + 1

        node_indices = torch.arange(self.num_nodes, device=self.device).repeat(batch.max().item() + 1)
        #print("Shape of node indices: ", node_indices.shape)

        # Get node index embeddings
        index_embeds = self.node_index_embedding(node_indices)  # [total_nodes, embedding_dim]
        #print("Shape of index embeddings: ", index_embeds.shape)

        # Concatenate raw features with index embeddings
        x = torch.cat([x, index_embeds], dim=1)  # Shape: [total_nodes, num_features + embedding_dim]
        #print("Concatenation of input after embedding", x.shape)

        #print("Shape before feature FC: ", x.shape)
        # Embedding node features if needed - input
        x = self.feature_fc(x)
        #print("Feature FC after shape: ", x.shape)

        #print("Shape after feature FC: ", x.shape)

        # Input GATConv
        x = self.input_gat_conv(x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x)
        #print("Shape x after input GATConv: ", x.shape)

        # Local graph feature extraction, using graph attention layers
        for gat_conv in self.gatconv_layers:
            x = gat_conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            #print("Shape x after GATConv: ", x.shape)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]
        #print("Shape x after decoder squeeze: ", x.shape)

        x = x.reshape(batch_size,self.num_nodes,-1)
        #print("Shape x after batch reshape: ", x.shape)

        # Global attention applied through transformer
        for decoder in self.transformer_decoder:
            x, attn_weights = decoder(query=x,
                                      key=x,
                                      value=x)
            #print("Shape x after decoder layer: ", x.shape)
            #print(x[0, :])
            #print(x[0, :])

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)
        #print("Shape x after decoder squeeze: ", x.shape)

        x = self.node_proj_FC(x)
        #print("Shape after node projection", x.shape)

        batch_size = int(batch.max().item())+1
        x = x.reshape(batch_size, -1)
        #print("Shape after batch projection", x.shape)

        # Final fully connected layer for the output
        x = self.out(x)
        #print("Shape of output: ", x.shape)

        return x

#TODO Not used


# TODO TI GNN GCN - PMU_caseB or conventional
class TI_GCNNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(TI_GCNNoEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features)
        self.conv1 = GCNConv(num_features, 64)  # No edge_dim
        self.conv2 = GCNConv(64, 16)  # No edge_dim
        # self.conv3 = GATConv(32 * heads, 8, heads=heads, concat=True)
        # self.conv4 = GATConv(8  * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16  * heads, 4, heads=heads, concat=True)

        # self.fc1 = torch.nn.Linear(8*heads, 32)
        # self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)))

        self.fc1 = torch.nn.Linear(16, 32)
        # self.fc2 = torch.nn.Linear(64, 32)

        # Fully connected layer for classification
        self.fc3 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        # If no node features exist, initialize dummy features
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, mask, batch = data.x.to(self.device), data.edge_index.to(self.device), data.mask.to(
            self.device), data.batch.to(self.device)

        # First GAT layer
        x = self.conv1(x=x, edge_index=edge_index)
        x = F.leaky_relu(x)
        # x = self.dropout(x)

        # Second GAT layer
        x = self.conv2(x=x, edge_index=edge_index)
        x = F.leaky_relu(x)
        # x = self.dropout(x)
        # x = x.view(1, -1) # [1056, 16]
        # Global mean pooling
        # x = global_mean_pool(x, batch)

        # x = self.attn_pool(x, batch)
        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)

        return x







