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


class SparseAwareGATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, bias: bool = True):
        super(SparseAwareGATConv, self).__init__(aggr='add')  # "Add" aggregation (standard in GAT)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformation for node features
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        # Bias term
        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        else:
            self.register_parameter('bias', None)

        # Learnable small feature vector for zero nodes
        self.zero_placeholder = Parameter(torch.zeros(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index):
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Identify zero-feature nodes
        zero_mask = (x.abs().sum(dim=1) == 0)

        # Replace zero nodes with learnable placeholder
        x = x.clone()
        x[zero_mask] = self.zero_placeholder

        # Apply linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # Compute attention scores
        x_i = x[edge_index[0]]  # Source nodes
        x_j = x[edge_index[1]]  # Target nodes
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Apply masking: Ignore edges where **both nodes** are zero
        valid_edges = ~(zero_mask[edge_index[0]] & zero_mask[edge_index[1]])
        alpha[~valid_edges] = float('-inf')  # Assign -inf to softmax to ignore

        # Compute softmax attention weights
        alpha = softmax(alpha, edge_index[1])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Perform message passing
        out = self.propagate(edge_index, x=x, alpha=alpha)

        # If concat, reshape; otherwise, take mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Apply bias if available
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


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
