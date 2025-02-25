import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP
from model_components import *
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


#TODO Simple GATConv stacking
class SE_GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4):
        super(SE_GATNoEdgeAttrs, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

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
        self.conv1 = GATConv(embedding_dim, GATConv1_dim, heads=heads, concat=True)
        self.conv2 = GATConv(GATConv1_dim * heads, GATConv2_dim, heads=heads, concat=True)

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


# TODO Sparse GAT layer
class SparseGATConvModel(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4):
        super(SparseGATConvModel, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = SparseAwareGATConv(in_channels=num_features, out_channels=64, heads=heads, concat=True)
        self.conv2 = SparseAwareGATConv(in_channels=64 * heads, out_channels=16, heads=heads, concat=True)
        # self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        # self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1),
                                 dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x







