import torch
import torch.nn as nn

from attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(0.01)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm_1(x + self.drop_out(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.drop_out(ff_output))
        return x