import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, d_context, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Project context to d_model dimension
        self.project_context = nn.Sequential(
            nn.Linear(d_context, d_context),
            nn.GELU(),
            nn.Linear(d_context, d_model)
        )
        
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, hidden_states, context_hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        context_seq_length = context_hidden_states.shape[1]

        if attention_mask is None:
            attention_mask = torch.zeros((batch_size, seq_length, context_seq_length), device=hidden_states.device)
        attention_mask = attention_mask.unsqueeze(1)  # Add heads dimension

        # Project context hidden states
        context_hidden_states = self.project_context(context_hidden_states)
        
        # Calculate queries, keys, and values
        Q = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        K = self.key(context_hidden_states).view(batch_size, context_seq_length, self.num_heads, self.d_head).transpose(1, 2)
        V = self.value(context_hidden_states).view(batch_size, context_seq_length, self.num_heads, self.d_head).transpose(1, 2)
        
        attention_scores = torch.einsum('bnqd,bnkd->bnqk', Q, K) / math.sqrt(self.d_head)
        attention_scores += attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.einsum('bnqk,bnkd->bnqd', attention_probs, V).transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        # Apply output projection
        output = self.out_proj(context)
        return output