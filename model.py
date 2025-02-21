import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, GINConv, GraphNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, GINConv, GraphNorm

#TODO To define the Transformer Encoder and Transformer Decoder layers manually without relying
#      on TransformerEncoderLayer and TransformerDecoderLayer from PyTorch, we can break down the individual
#      components of a Transformer layer, which are:
#      Multihead Self-Attention: Each token (node feature in our case) is transformed by attending to every other
#      token in the sequence (graph) via a weighted sum (attention mechanism).
#      Feedforward Neural Network (FFN): After the attention step, a position-wise fully connected feedforward network
#      is applied to each token. Normalization: After each sub-layer (self-attention and FFN), layer normalization
#      is applied to stabilize training. Residual Connections: The output of each sub-layer is added back to the input,
#      creating residual connections.
#TODO Simple GATConv stacking
class SE_GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4):
        super(SE_GATNoEdgeAttrs, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 32, heads=heads, concat=True)
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)
        #self.conv4 = GATConv(8 * heads, 4, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(32 * heads, output_dim)

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

#TODO Encoder-Decoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(EncoderLayer, self).__init__()

        # Multihead Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Activation
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, src):
        # Self-attention
        tgt, _ = self.self_attention(src, src, src)  # Q=K=V=src

        output = self.linear1(tgt)
        output = self.activation1(output)
        output = self.linear2(output)
        output = self.activation2(output)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(DecoderLayer, self).__init__()

        #TODO Used for best model so far
        # Multihead Attention Layer (Self-Attention)
        # self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Multihead Attention Layer (Encoder-Decoder Attention)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Linear functions
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Activation functions
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()


    def forward(self, tgt, memory):
        # Self-attention (decoder layer self-attention)
        #tgt, _ = self.self_attn(tgt, tgt, tgt)

        # Multihead attention
        attn_output, _ = self.multihead_attn(tgt, memory, memory)

        output = self.linear1(attn_output)
        output = self.activation1(output)
        output = self.linear2(output)
        output = self.activation2(output)

        return output

class GATEncoderDecoder(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4, num_encoder_layers=1,
                 num_decoder_layers=1, GATConv1_dim=64, GATConv2_dim=16, ff_hid_dim=32):
        super(GATEncoderDecoder, self).__init__()

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Layers (Graph Attention)
        self.conv1 = GATConv(embedding_dim, GATConv1_dim, heads=heads, concat=True)
        self.conv2 = GATConv(GATConv1_dim * heads, GATConv2_dim, heads=heads, concat=True)

        # Custom Transformer Encoder Layer
        self.transformer_encoder = nn.ModuleList([
            EncoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_encoder_layers)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            DecoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_decoder_layers)
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
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)

        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, x)  # Decoder attends to encoder output only

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)

        # Apply global pooling: aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Final fully connected layer for the output
        x = self.fc(x)

        return x

#TODO Transfomer with self and cross-attention
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        #self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attention with Residual Connection
        attn_output, _ = self.self_attention(src, src, src)

        # Feedforward Network with Residual Connection
        output = self.linear1(src)
        output = self.activation1(output)
        output = self.linear2(output)
        output = self.activation2(output)  # Residual Connection 2

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        #self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        #self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-Attention
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt)

        # Encoder-Decoder Attention with Residual Connection
        attn_output, _ = self.multihead_attn(self_attn_output, memory, memory)

        # Feedforward Network with Residual Connection
        output = self.linear1(tgt)
        output = self.activation1(output)
        #output = self.dropout(output)
        output = self.linear2(output)
        output = self.activation2(output)

        return output

class GATTransformer(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=4, heads=4,
                 num_encoder_layers=1, num_decoder_layers=1, GATConv1_dim=64, GATConv2_dim=16, ff_hid_dim=32, dropout=0.1):
        super(GATTransformer, self).__init__()

        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Layers
        self.conv1 = GATConv(embedding_dim, GATConv1_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(GATConv1_dim * heads, GATConv2_dim, heads=heads, concat=True, dropout=dropout)

        # Transformer Encoder & Decoder
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model=GATConv2_dim * heads, nhead=heads, dim_feedforward=ff_hid_dim, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc = nn.Linear(GATConv2_dim * heads, output_dim)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding node features if needed
        if x is None:
            x = self.node_embedding(data.batch)
        elif self.feature_fc is not None:
            x = self.feature_fc(x)

        # Apply GAT layers
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        # Prepare for Transformer
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]

        # Transformer Encoder
        memory = x
        for encoder_layer in self.transformer_encoder:
            memory = encoder_layer(memory)

        # Transformer Decoder
        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, memory)

        x = x.squeeze(0)  # Remove batch dim
        x = global_mean_pool(x, batch)  # Aggregate node-level to graph-level

        return self.fc(x)  # Final Output


