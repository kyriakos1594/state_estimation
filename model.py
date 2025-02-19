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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multihead Attention Layer
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        attn_output, _ = self.self_attention(src, src, src)  # Q=K=V=src

        # Add & Normalize
        src = src + self.dropout1(attn_output)  # Residual connection
        src = self.norm1(src)  # Layer normalization

        # Feedforward Network
        ffn_output = self.ffn(src)

        # Add & Normalize
        src = src + self.dropout2(ffn_output)  # Residual connection
        src = self.norm2(src)  # Layer normalization

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Multihead Attention Layer (Self-Attention)
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Multihead Attention Layer (Encoder-Decoder Attention)
        self.encoder_decoder_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention (decoder layer self-attention)
        attn_output, _ = self.self_attention(tgt, tgt, tgt)

        # Add & Normalize
        tgt = tgt + self.dropout1(attn_output)  # Residual connection
        tgt = self.norm1(tgt)  # Layer normalization

        # Encoder-Decoder Attention
        attn_output, _ = self.encoder_decoder_attention(tgt, memory, memory)

        # Add & Normalize
        tgt = tgt + self.dropout2(attn_output)  # Residual connection
        tgt = self.norm2(tgt)  # Layer normalization

        # Feedforward Network
        ffn_output = self.ffn(tgt)

        # Add & Normalize
        tgt = tgt + self.dropout3(ffn_output)  # Residual connection
        tgt = self.norm3(tgt)  # Layer normalization

        return tgt

class HybridGATTransformer(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=32, heads=4, num_encoder_layers=2, num_decoder_layers=2, ff_hid_dim=256):
        super(HybridGATTransformer, self).__init__()

        # Node embedding layer (if needed, otherwise use raw features)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # GAT Layers (Graph Attention)
        self.conv1 = GATConv(embedding_dim, 32, heads=heads, concat=True)
        self.conv2 = GATConv(32 * heads, 16, heads=heads, concat=True)

        # Custom Transformer Encoder Layer
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=16 * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_encoder_layers)
        ])

        # Custom Transformer Decoder Layer
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model=16 * heads, nhead=heads, dim_feedforward=ff_hid_dim) for _ in range(num_decoder_layers)
        ])

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer for output (e.g., classification or regression)
        self.fc = nn.Linear(16 * heads, output_dim)

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
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # Prepare for Transformer by adding a batch dimension (1, batch_size, features)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, feature_dim]

        # Apply Transformer Encoder
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)

        # Apply Transformer Decoder
        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(x, x)  # Decoder attends to encoder output

        # Remove the batch dimension (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        x = x.squeeze(0)

        # Apply global pooling: aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Final fully connected layer for the output
        x = self.fc(x)

        return x

# Example usage
num_nodes = 33  # Number of nodes (for node embedding)
num_features = 4  # Number of features per node
output_dim = 33  # Number of output classes (or regression outputs)
embedding_dim = 8  # Embedding dimension for nodes
heads = 4  # Number of attention heads in GAT
num_encoder_layers = 2  # Number of Transformer encoder layers
num_decoder_layers = 2  # Number of Transformer decoder layers
ff_hid_dim = 256  # Feedforward network dimension in Transformer

model = HybridGATTransformer(
    num_nodes, num_features, output_dim, embedding_dim, heads, num_encoder_layers, num_decoder_layers, ff_hid_dim
)
