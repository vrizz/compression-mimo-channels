"""
Credits:
This code is based on the repository https://github.com/lucidrains/vit-pytorch.
We acknowledge and are grateful to these developers for keeping their code open source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class ResBlock(nn.Module):

    def __init__(self, obs_dim, fw_module):
        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)
        self.fw_module = fw_module

    def forward(self, x, **kwargs):
        return self.fw_module(self.norm(x), **kwargs)



class ResNet(nn.Module):

    def __init__(self, res_block):
        super().__init__()
        self.res_block = res_block

    def forward(self, x, **kwargs):
        return self.res_block(x, **kwargs) + x



class MLP(nn.Module):

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, x):
        return self.mlp(x)



class Attention(nn.Module):

    def __init__(self, obs_dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv_net = nn.Linear(obs_dim, obs_dim * 3, bias=False)
        self.out_net = nn.Linear(obs_dim, obs_dim)
        self.scale_factor = obs_dim ** (-0.5)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv_net(x)
        q, k, v = rearrange(qkv, 'b n (c h d) -> c b h n d', c=3, h=self.heads)
        att_scores = F.softmax(torch.einsum('bhid,bhjd->bhij', q, k) * self.scale_factor, dim=-1)
        self_att = torch.einsum('bhij,bhjd->bhid', att_scores, v)
        return self.out_net(rearrange(self_att, 'b h n d -> b n (h d)'))



class Transformer(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResNet(ResBlock(obs_dim, Attention(obs_dim, heads))),
                        ResNet(ResBlock(obs_dim, MLP(obs_dim, hidden_dim)))
                    ]
                )
            )

    def forward(self, x):
        for att_block, mlp_block in self.layers:
            x = att_block(x)
            x = mlp_block(x)
        return x