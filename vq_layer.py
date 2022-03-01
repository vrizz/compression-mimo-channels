"""
Credits:
This code is based on the repository https://github.com/zalandoresearch/pytorch-vq-vae.
We acknowledge and are grateful to these developers for keeping their code open source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def _paired_dist(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    dist[dist < 0] = 0
    return dist



def _smoothing_func(x, epsilon=1e-5):
    d = x.shape[0]
    n = x.data.sum()
    return (x + epsilon) / (n + d * epsilon) * n



def _moving_average(x_mov, x_new, decay):
    return x_mov * decay + (1 - decay) * x_new



class VQLayer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, ema=False, decay=0.0):
        super(VQLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.ema = ema
        self.decay = decay
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.register_buffer('cluster_size', torch.zeros(self.num_embeddings))
        self.register_buffer('unnormalized_weights', torch.randn((self.num_embeddings, self.embedding_dim)))

        if ema is not True:
            self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        else:
            self.codebook.weight.data.normal_()

    def forward(self, x):
        x_shape = x.shape
        x_temp = x.view(-1, self.embedding_dim)
        n, _ = x_temp.shape
        dist = _paired_dist(x_temp, self.codebook.weight)
        index = dist.argmin(dim=1, keepdim=True)
        indicator_mat = torch.zeros(n, self.num_embeddings, device=x.device).scatter_(1, index, 1)
        z = torch.mm(indicator_mat, self.codebook.weight).view(x_shape)

        commitment_loss = F.mse_loss(z.detach(), x)

        avg_probs = indicator_mat.mean(dim=0)
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()

        if self.ema is not True:
            quantization_loss = F.mse_loss(z, x.detach())
            z = x + (z - x).detach()
            return z, quantization_loss, commitment_loss, perplexity
        else:
            if self.training:
                self.cluster_size = _moving_average(self.cluster_size, indicator_mat.sum(dim=0), self.decay)
                self.cluster_size = _smoothing_func(self.cluster_size)
                sum_u = torch.mm(indicator_mat.t(), x_temp.data)
                self.unnormalized_weights = _moving_average(self.unnormalized_weights, sum_u, self.decay)
                self.codebook.weight = nn.Parameter(self.unnormalized_weights / self.cluster_size.unsqueeze(1))
            z = x + (z - x).detach()
            return z, commitment_loss, perplexity