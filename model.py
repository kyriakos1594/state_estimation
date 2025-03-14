import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, global_max_pool, GlobalAttention
from model_components import *
from customGATConv import *
from config_file import *
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
        self.fc2 = nn.Linear(self.input_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)
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

#TODO TI GNN GAT - Only PMU_caseA
class TI_GATWithEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes, edge_attr_dim, heads=4):
        super(TI_GATWithEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        self.conv1  = GATConv(num_features, 32, heads=heads, concat=True, edge_dim=edge_attr_dim)  # First GAT layer with edge features
        self.conv2  = GATConv(32 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Second GAT layer
        #self.conv3  = GATConv(32 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Third GAT layer
        #self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc     = torch.nn.Linear(8 * heads, num_classes)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        batch = batch.to(self.device)

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index, edge_attr)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        #x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x #F.log_softmax(x, dim=1)

#TODO TI GNN GAT - PMU_caseB or conventional
class TI_GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_gat_layers=6, gat_dim=16, heads=4):
        super(TI_GATNoEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features) - Input
        self.input_conv = GATConv(num_features, gat_dim, heads=heads, concat=True)  # No edge_dim

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(gat_dim * heads, gat_dim, heads=heads, concat=True) for _ in range(num_gat_layers)
        ])

        self.fc1 = torch.nn.Linear(gat_dim * heads, 2 * gat_dim)

        # Fully connected layer for classification
        self.fc2 = torch.nn.Linear(2 * gat_dim, num_classes)

    def forward(self, data):
        # If no node features exist, initialize dummy features
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, mask, batch = data.x.to(self.device), data.edge_index.to(self.device), data.mask.to(self.device), data.batch.to(self.device)

        # First GAT layer
        x = self.input_conv(x, edge_index)
        x = F.relu(x)

        # Second GAT layer
        for gatconv_layer in self.GATConv_layers:
            x = gatconv_layer(x, edge_index)
            x = F.relu(x)

        #x = self.attn_pool(x, batch)
        x = global_max_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x

class TI_MultipleGCNNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes, heads=4):
        super(TI_MultipleGCNNoEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features)
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3  = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.conv6 = GCNConv(32, 32)
        self.conv7 = GCNConv(32, 32)
        self.conv8 = GCNConv(32, 32)
        #self.conv9 = GCNConv(32, 32)
        #self.conv10 = GCNConv(32, 32)
        #self.conv11 = GCNConv(8 * heads, 8)
        #self.conv12 = GCNConv(8 * heads, 8)
        #self.conv13 = GCNConv(8 * heads, 8)
        #self.conv14 = GCNConv(8 * heads, 8)

        #self.fc1 = torch.nn.Linear(8*heads, 32)
        #self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(16 * heads, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)))

        self.fc1 = torch.nn.Linear(32, 32)
        #self.fc2 = torch.nn.Linear(64, 32)

        # Fully connected layer for classification
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        # If no node features exist, initialize dummy features
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, mask, batch = data.x.to(self.device), data.edge_index.to(self.device), data.mask.to(self.device), data.batch.to(self.device)

        #constant_edges = [i for i in range(NUM_BRANCHES) if (i not in [32,33,34,6,10,27])]
        #print(constant_edges)
        #edge_index = edge_index[:, constant_edges]

        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)
        #x = x.view(1, -1) # [1056, 16]
        # Global mean pooling
        #x = global_mean_pool(x, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        x = self.conv6(x, edge_index)
        x = F.relu(x)

        x = self.conv7(x, edge_index)
        x = F.relu(x)
        x = self.conv8(x, edge_index)
        x = F.relu(x)
        #x = self.conv9(x, edge_index)
        #x = F.relu(x)
        #x = self.conv10(x, edge_index)
        #x = F.relu(x)
        #x = self.conv11(x, edge_index)
        #x = F.relu(x)
        #x = self.conv12(x, edge_index)
        #x = F.relu(x)
        #x = self.conv13(x, edge_index)
        #x = F.relu(x)
        #x = self.conv14(x, edge_index)
        #x = F.relu(x)
        #x = self.attn_pool(x, batch)
        x = global_max_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x

#TODO TI GNN GCN - PMU_caseB or conventional
class TI_GCNNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(TI_GCNNoEdgeAttrs, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Graph Attention layers (without edge features)
        self.conv1 = GCNConv(num_features, 64)  # No edge_dim
        self.conv2 = GCNConv(64, 16)  # No edge_dim
        #self.conv3 = GATConv(32 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8  * heads, 4, heads=heads, concat=True)
        #self.conv4 = GATConv(16  * heads, 4, heads=heads, concat=True)

        #self.fc1 = torch.nn.Linear(8*heads, 32)
        #self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)))

        self.fc1 = torch.nn.Linear(16, 32)
        #self.fc2 = torch.nn.Linear(64, 32)

        # Fully connected layer for classification
        self.fc3 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        # If no node features exist, initialize dummy features
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, mask, batch = data.x.to(self.device), data.edge_index.to(self.device), data.mask.to(self.device), data.batch.to(self.device)

        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        #x = x.view(1, -1) # [1056, 16]
        # Global mean pooling
        #x = global_mean_pool(x, batch)

        #x = self.attn_pool(x, batch)
        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)

        return x

#TODO TI Transformer - PMU_caseB or conventional
class TI_TransformerNoEdges(torch.nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, GATConv_layers, GATConv_dim, embedding_dim=4, heads=4, dec_layers=1, ff_hid_dim=32):
        super(TI_TransformerNoEdges, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Convolution Layers (Graph Attention) - Input
        self.input_conv_layer = GATConv(embedding_dim, GATConv_dim, heads=heads, concat=True)

        # GAT Convolution Layers (Graph Attention) - Stacking to retrieve features n-hops away
        self.GATConv_layers = nn.ModuleList([
            GATConv(GATConv_dim * heads, GATConv_dim, heads=heads, concat=True) for _ in range(GATConv_layers)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransfromerDecoderLayer2(d_model=GATConv_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(dec_layers)
        ])

        # Global attention mechanism, instead of max or mean pooling
        self.attn_pool = GlobalAttention(gate_nn = nn.Sequential(torch.nn.Linear(GATConv_dim * heads, ff_hid_dim),
                                                                 torch.nn.ReLU(),
                                                                 torch.nn.Linear(ff_hid_dim, 1)))

        #self.attn_pool = GlobalAttention(in_channels)

        # Fully connected layer for output (e.g., classification or regression)
        # Final output layer
        self.fc = nn.Linear(GATConv_dim * heads, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding node features if needed
        if x is None:
            x = self.node_embedding(data.batch)
        elif self.feature_fc is not None:
            x = self.feature_fc(x)

        # Apply GAT layers to process graph structure and features
        x = self.input_conv_layer(x, edge_index)
        x = F.leaky_relu(x)
        # x = self.dropout(x)

        for gat_conv_layer in self.GATConv_layers:
            x = gat_conv_layer(x, edge_index)
            x = F.leaky_relu(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]

        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, x)  # Decoder attends to encoder output only

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)
        # Apply global pooling: aggregate node-level features into graph-level features
        x = self.attn_pool(x, batch)

        # Final fully connected layer for the output
        x = self.fc(x)

        return x


#TODO Simple GATConv stacking - No Mask
class SE_GATNoEdgeAttrsNoMask(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=False, heads=4):
        super(SE_GATNoEdgeAttrsNoMask, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = GATConv(num_features, 16, heads=heads, concat=True)
        self.conv2 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv3 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv4 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv5 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv6 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv7 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv8 = GATConv(16 * heads, 16, heads=heads, concat=True)


        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=False

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        x = self.conv6(x, edge_index)
        x = F.relu(x)

        x = self.conv7(x, edge_index)
        x = F.relu(x)

        x = self.conv8(x, edge_index)
        x = F.relu(x)


        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

#TODO Simple GATConv stacking - Masked
class SE_GATNoEdgeAttrsMask(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=True, heads=4):
        super(SE_GATNoEdgeAttrsMask, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=True

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

# TODO Only Source node attention calculation - No Mask
class SE_OnlySourceNodeAttentionNoMaskGATConvModel(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=False, heads=4):
        super(SE_OnlySourceNodeAttentionNoMaskGATConvModel, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = CustomGATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = CustomGATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=False

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

# TODO Only Source node attention calculation - Masked
class SE_OnlySourceAttentionMaskGATConvModel(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=False, heads=4):
        super(SE_OnlySourceAttentionMaskGATConvModel, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = CustomGATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = CustomGATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=True

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

# TODO Only Source node attention calculation for first node - No Mask
class SE_FirstSourceNodeAttentionNoMaskGATConvModel(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=False, heads=4):
        super(SE_FirstSourceNodeAttentionNoMaskGATConvModel, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = CustomGATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=False

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

# TODO Only Source node attention calculation - Masked
class SE_FirstSourceAttentionMaskGATConvModel(torch.nn.Module):
    def __init__(self, num_features, output_dim, mask=False, heads=4):
        super(SE_FirstSourceAttentionMaskGATConvModel, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = CustomGATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        #self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

        self.mask=True

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        if self.mask == False:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch, edge_mask = data.x, data.edge_index, data.batch, data.edge_mask

            active_edges = edge_index*edge_mask
            edge_index = active_edges

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Third GAT layer with edge attributes
        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

#TODO Selective Mask GATConvModel
class SE_GATNoEdgeAttrsSelectiveMask(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4):
        super(SE_GATNoEdgeAttrsSelectiveMask, self).__init__()

        # GAT Layers
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)

        # Fully connected output layer
        self.fc = torch.nn.Linear(16 * heads, output_dim)

    def filter_edges(self, x, edge_index, original_edge_index=None, second_hop=False):
        """
        Filters edges based on updated node features, keeping only second-hop neighbors
        relative to the initial edge structure.
        """
        if second_hop:
            # Step 1: Find the second-hop neighbors relative to the original edges
            first_hop_neighbors = edge_index[1]  # Destination nodes of original edges
            second_hop_neighbors = edge_index[:, first_hop_neighbors]  # Second-hop neighbors

            # Step 2: Keep edges only between second-hop neighbors
            # We will need to check if destination nodes of the edges are second-hop neighbors
            second_hop_mask = torch.isin(edge_index[1], second_hop_neighbors)
            edge_index = edge_index[:, second_hop_mask]

        else:
            # Normal filtering, only keeping edges where at least one node has a non-zero feature
            node_mask = (x.abs().sum(dim=1) > 0)  # True for nodes with non-zero features
            edge_mask = node_mask[edge_index[0]] | node_mask[
                edge_index[1]]  # Keep edges if either node has non-zero features
            edge_index = edge_index[:, edge_mask]

        return x, edge_index  # Keep node features unchanged

    def forward(self, data):
        # Initialize dummy features if `data.x` is None
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)

        x, edge_index, batch = data.x, data.edge_index, data.batch
        main_edge_index = edge_index

        # Filter edges BEFORE the first GATConv layer
        #x, edge_index = self.filter_edges(x, edge_index)
        edge_index = main_edge_index[[17, 26], :]
        # First GAT Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #print("First layer edges (filtered): ", edge_index)

        # **Find 2nd-hop neighbors after the first layer**
        #x, edge_index = self.filter_edges(x, edge_index, original_edge_index=data.edge_index, second_hop=True)
        edge_index = main_edge_index[[15, 16, 17, 24, 25, 26], :]
        # Second GAT Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #print("Second layer edges (filtered): ", edge_index)

        #import time
        #time.sleep(10)

        # Graph-level representation
        x = global_mean_pool(x, batch)

        # Fully connected output layer
        x = self.fc(x)

        return x



#TODO TRANSFORMERS SE

#TODO Only Decoder Transformer - only self attention
class GATTransfomerOnlyDecoder(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4, num_encoder_layers=1,
                 num_decoder_layers=1, GATConv1_dim=64, GATConv2_dim=16, ff_hid_dim=32):
        super(GATTransfomerOnlyDecoder, self).__init__()

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Layers (Graph Attention)
        self.conv1 = GATConv(embedding_dim, 16, heads=heads, concat=True)
        self.conv2 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv3 = GATConv(16 * heads, 16, heads=heads, concat=True)
        self.conv4 = GATConv(16 * heads, 16, heads=heads, concat=True)

        # Custom Transformer Encoder Layer
        #self.transformer_encoder = nn.ModuleList([
        #    TransfromerDecoderLayer1(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_encoder_layers)
        #])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransfromerDecoderLayer2(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_decoder_layers)
        ])

        # Dropout layer for regularization
        #self.dropout = nn.Dropout(0.3)

        # Fully connected layer for output (e.g., classification or regression)
        self.fc = nn.Linear(GATConv2_dim * heads, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding node features if needed
        if x is None:
            x = self.node_embedding(data.batch)
        elif self.feature_fc is not None:
            x = self.feature_fc(x)

        # Apply GAT layers to process graph structure and features
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)


        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]

        #TODO Old architecture decode(x, x)
        #Apply Transformer Encoder
        #for encoder_layer in self.transformer_encoder:
        #    x = encoder_layer(x)

        # Apply Transformer Decoder
        #for decoder_layer in self.transformer_decoder:
        #    x = decoder_layer(x, x)  # Decoder attends to encoder output

        #TODO Old Transformer
        #memory = x
        #for encoder_layer in self.transformer_encoder:
        #    memory = encoder_layer(memory)

        #for decoder_layer in self.transformer_decoder:
        #    x = decoder_layer(x, memory)  # Decoder attends to encoder output and decoder's initial signal input

        #TODO Encode and decode input signal with self attention and multi attention
        #for encoder_layer in self.transformer_encoder:
        #    x = encoder_layer(x)

        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, x)  # Decoder attends to encoder output only

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)

        # Apply global pooling: aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Final fully connected layer for the output
        x = self.fc(x)

        return x

#TODO Transfomer Encoder-Decoder with self and cross-attention
class GATTransformerEncoderDecoder(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4, num_encoder_layers=1,
                 num_decoder_layers=1, GATConv1_dim=64, GATConv2_dim=16, ff_hid_dim=32):
        super(GATTransformerEncoderDecoder, self).__init__()

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Layers (Graph Attention)
        self.conv1 = GATConv(embedding_dim, GATConv1_dim, heads=heads, concat=True)
        self.conv2 = GATConv(GATConv1_dim * heads, GATConv2_dim, heads=heads, concat=True)

        # Custom Transformer Encoder Layer
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_encoder_layers)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_decoder_layers)
        ])

        # Dropout layer for regularization
        #self.dropout = nn.Dropout(0.3)

        # Fully connected layer for output (e.g., classification or regression)
        self.fc = nn.Linear(GATConv2_dim * heads, output_dim)

    def forward(self, data):
        # Extract node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding node features if needed
        if x is None:
            x = self.node_embedding(data.batch)
        elif self.feature_fc is not None:
            x = self.feature_fc(x)

        # Apply GAT layers to process graph structure and features
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]

        #TODO Old architecture decode(x, x)
        #Apply Transformer Encoder
        #for encoder_layer in self.transformer_encoder:
        #    x = encoder_layer(x)

        # Apply Transformer Decoder
        #for decoder_layer in self.transformer_decoder:
        #    x = decoder_layer(x, x)  # Decoder attends to encoder output

        #TODO Old Transformer
        #memory = x
        #for encoder_layer in self.transformer_encoder:
        #    memory = encoder_layer(memory)

        #for decoder_layer in self.transformer_decoder:
        #    x = decoder_layer(x, memory)  # Decoder attends to encoder output and decoder's initial signal input

        #TODO Encode and decode input signal with self attention and multi attention
        memory = x
        for encoder_layer in self.transformer_encoder:
            memory = encoder_layer(memory)

        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, memory)  # Decoder attends to encoder output only

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)

        # Apply global pooling: aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Final fully connected layer for the output
        x = self.fc(x)

        return x

# TODO Encoder and decoder layers with GATConv layers
class GATConvTransformer(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4, ff_hid_dim=32):
        super(GATConvTransformer, self).__init__()

        self.node_embedding = nn.Embedding(num_features, embedding_dim)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # Single Encoder & Decoder
        self.encoder = GATConvTransformerEncoderLayer(d_model=embedding_dim, nhead=heads, num_features=num_features,
                                                      dim_feedforward=ff_hid_dim)
        self.decoder = GATConvTransformerDecoderLayer(d_model=embedding_dim, nhead=heads, num_features=num_features,
                                                      dim_feedforward=ff_hid_dim)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding node features if needed
        if x is None:
            x = self.node_embedding(batch)
        elif self.feature_fc is not None:
            x = self.feature_fc(x)  # [NUM_NODES, ]

        # Transformer Encoder
        memory = self.encoder(x, edge_index)

        # Transformer Decoder
        x = self.decoder(x, memory, x, edge_index)

        # Global pooling & final output
        x = global_mean_pool(x, batch)

        return self.fc(x)








