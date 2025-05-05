import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransfromerDecoderLayer1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(TransfromerDecoderLayer1, self).__init__()

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

class TransfromerDecoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(TransfromerDecoderLayer2, self).__init__()

        #TODO Used for best model so far
        # Multihead Attention Layer (Self-Attention)
        # self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Multihead Attention Layer (Encoder-Decoder Attention)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Linear functions
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Activation functions
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()


    def forward(self, tgt, memory, key_padding_mask=None):
        # Self-attention (decoder layer self-attention)
        #tgt, _ = self.self_attn(tgt, tgt, tgt)
        #print(tgt.shape, memory.shape, memory.shape)
        # Multihead attention
        attn_output, _ = self.multihead_attn(tgt, memory, memory, key_padding_mask=key_padding_mask)
        #print(attn_output.shape)

        #import time
        #time.sleep(100)

        output = self.linear1(attn_output)
        output = self.activation1(output)
        output = self.linear2(output)
        output = self.activation2(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(TransformerEncoderLayer, self).__init__()

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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128):
        super(TransformerDecoderLayer, self).__init__()

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


class GATConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_features, dim_feedforward=128):
        super(GATConvTransformerEncoderLayer, self).__init__()

        # GAT for self-attention inside the encoder
        self.gat1 = GATConv(num_features, d_model, heads=nhead, concat=True)
        self.gat2 = GATConv(nhead * d_model, d_model, heads=nhead, concat=True)  # Additional GATConv

        # Feedforward network
        self.linear1 = nn.Linear(nhead * d_model, nhead * dim_feedforward)
        self.activation1 = nn.ReLU()

    def forward(self, src, edge_index):
        # First GAT-based self-attention
        src = self.gat1(src, edge_index)
        src = F.leaky_relu(src)

        # Second GAT-based self-attention
        src = self.gat2(src, edge_index)
        src = F.leaky_relu(src)

        # Feedforward
        output = self.linear1(src)
        output = self.activation1(output)

        return output

class GATConvTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_features, dim_feedforward=128):
        super(GATConvTransformerDecoderLayer, self).__init__()

        # GAT for self-attention inside the decoder
        self.self_gat = GATConv(num_features, d_model, heads=nhead, concat=True)

        # MultiheadAttention for cross-attention with encoder output
        self.cross_attn = nn.MultiheadAttention(embed_dim=nhead * d_model, num_heads=nhead, dropout=0.1)

        # Feedforward network
        self.linear1 = nn.Linear(nhead * d_model, dim_feedforward)
        self.activation1 = nn.ReLU()

    def forward(self, tgt, memory, original_signal, edge_index):
        # Self-attention using GAT
        tgt = self.self_gat(tgt, edge_index)
        tgt = F.leaky_relu(tgt)

        # Attention-based fusion of encoder output and original signal
        # Project both memory (encoder output) and original_signal (input features) into the same space
        query = tgt  # Using target as query
        key = memory  # Encoder output as key
        value = original_signal  # Using original signal as value

        # Apply MultiheadAttention to fuse encoder output and original signal
        attn_output, _ = self.cross_attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))

        # Remove sequence dimension (because the output of multihead attention has shape [1, num_nodes, d_model])
        fused_output = attn_output.squeeze(0)

        # Feedforward network
        output = self.linear1(fused_output)
        output = self.activation1(output)

        return output
