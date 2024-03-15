import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional
    
    
class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, d_context, num_heads=4):
        super(MultiHeadCrossAttentionLayer, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.project_aug = nn.Linear(d_context, d_model)
        # Learnable weight matrices
        self.WQ = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.WK = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.WV = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.WO = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        
    def forward(self, hidden_states, context_hidden_states, attention_mask:Optional[torch.Tensor]=None):

        batch_size, seq_length, hidden_dim = hidden_states.shape
        context_seq_length = context_hidden_states.shape[1]
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length, context_seq_length), device=device)
        
        # Project the context hidden states
        context_hidden_states = self.project_aug(context_hidden_states)
        
        # Calculate the queries, keys and values
        Q = torch.matmul(hidden_states, self.WQ)
        K = torch.matmul(context_hidden_states, self.WK)
        V = torch.matmul(context_hidden_states, self.WV)
 
        # Reshape the queries, keys and values
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_head)
        K = K.view(batch_size, context_seq_length, self.num_heads, self.d_head)
        V = V.view(batch_size, context_seq_length, self.num_heads, self.d_head)

        # print(Q.shape, K.shape, V.shape)
        attention = torch.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.d_head)

        # Apply the attention mask if using _4d_attention_mask from hf, just add the mask to the attention (0 is for tokens to attend to, -inf is for tokens to ignore)
        attention = F.softmax(attention, dim=-1)
        context = torch.einsum('bhqk,bkhd->bhqd', attention, V)
        context = context.view(batch_size, seq_length, self.d_model)

        
        # Project the context
        context = torch.matmul(context, self.WO)
        
        return context
        

