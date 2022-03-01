from transformer import *
from vq_layer import *


class SLVit(nn.Module):

    def __init__(self, n_patches=16, dim=128, heads=8, hidden_dim=128):
        super().__init__()
        self.embedding_proj = nn.Linear(dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, dim))
        self.transformer = Transformer(obs_dim=dim, heads=heads, hidden_dim=hidden_dim, n_layers=1)

    def forward(self, x):
        x = self.embedding_proj(x)
        x += self.pos_embedding
        x = self.transformer(x)
        return x



class Encoder(nn.Module):

    def __init__(self, dims_enc=[128, 64, 32], heads_enc=[8, 8, 32], initial_scale_factor=0.5,
                 p1=2, p2=2, hidden_dim=128, mode="linear", input_patches=5):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.initial_scale_factor = initial_scale_factor
        self.transf_enc = nn.ModuleList(
            [
                SLVit(n_patches=input_patches, dim=dim, heads=heads, hidden_dim=hidden_dim) for dim, heads in
                zip(dims_enc, heads_enc)
            ]
        )
        self.mode = mode

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p1, p2=self.p2)
        for i, t in enumerate(self.transf_enc):
            x = t(x)
            # downsample 1) N x 16 x x -> N x 16 x 64
            # downsample 2) N x 16 x 64 -> N x 16 x 32
            if i == 0:
                x = F.interpolate(x, scale_factor=self.initial_scale_factor, mode=self.mode)
            if i == 1:
                x = F.interpolate(x, scale_factor=0.5, mode=self.mode)
        return x



class Decoder(nn.Module):

    def __init__(self, dims_dec=[128, 64, 32], heads_dec=[8, 8, 32], initial_scale_factor=0.5, p1=2, p2=2, h=8,
                 hidden_dim=128, mode="linear", input_patches=5):
        super().__init__()
        self.transf_dec = nn.ModuleList(
            [
                SLVit(n_patches=input_patches, dim=dim, heads=heads, hidden_dim=hidden_dim) for dim, heads in
                zip(dims_dec, heads_dec)
            ]
        )
        self.mode = mode
        self.p1 = p1
        self.p2 = p2
        self.h = h
        self.initial_scale_factor = initial_scale_factor

    def forward(self, x):
        for i, td in enumerate(self.transf_dec):
            x = td(x)
            # upsample 1) N x 16 x 8 -> N x 16 x 64
            # upsample 2) N x 16 x 64 -> N x 16 x N
            if i == 0:
                x = F.interpolate(x, scale_factor=2, mode=self.mode)
            if i == 1:
                x = F.interpolate(x, scale_factor=self.initial_scale_factor ** (-1), mode=self.mode)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p1, p2=self.p2, h=self.h)
        return x



class SLViTAE(nn.Module):

    def __init__(self, dims_enc=[128, 64, 32], heads_enc=[8, 8, 32], overall_dim=2048, p1=2, p2=2, hidden_dim=128,
                 mode="linear"):
        super().__init__()
        patch_size = p1 * p2 * 2
        self.h = int(16 / p1)  # height of the image divided by the size of the patch
        input_patches = overall_dim // patch_size
        dims_enc[0] = patch_size
        self.input_patches = input_patches
        self.p1 = p1
        self.p2 = p2
        dims_enc[1] = int(dims_enc[0] / 2)
        dims_enc[2] = int(dims_enc[1] / 2)

        if dims_enc[0] == 128:
            self.initial_scale_factor = 0.5

        elif dims_enc[0] == 256:
            self.initial_scale_factor = 0.5

        elif dims_enc[0] == 64:
            self.initial_scale_factor = 0.5

        for i in range(len(dims_enc)):
            assert dims_enc[i] % heads_enc[i] == 0
            assert dims_enc[i] >= heads_enc[i]

        self.transf_enc = Encoder(
            dims_enc=dims_enc,
            heads_enc=heads_enc,
            initial_scale_factor=self.initial_scale_factor,
            p1=p1,
            p2=p2,
            hidden_dim=hidden_dim,
            mode=mode,
            input_patches=input_patches
        )
        dims_dec = dims_enc.copy()
        dims_dec.reverse()
        heads_dec = heads_enc.copy()
        heads_dec.reverse()
        self.transf_dec = Decoder(
            dims_dec=dims_dec,
            heads_dec=heads_dec,
            initial_scale_factor=self.initial_scale_factor,
            p1=p1,
            p2=p2,
            h=self.h,
            hidden_dim=hidden_dim,
            mode=mode,
            input_patches=input_patches
        )
        self.mode = mode

    def forward(self, x):
        x = self.transf_enc.forward(x)
        x = self.transf_dec.forward(x)
        return x



class SLViTAEQuant(SLViTAE):

    def __init__(self, dims_enc=[128, 64, 32], heads_enc=[8, 8, 32], overall_dim=2048, p1=2, p2=2, hidden_dim=128,
                 mode="linear", num_embeddings=64, embedding_dim=512, decay=0.0):
        super().__init__(dims_enc, heads_enc, overall_dim, p1, p2, hidden_dim, mode)

        if decay > 0.0:
            self.ema = True
            self.vq_layer = VQLayer(num_embeddings, embedding_dim, ema=True, decay=decay)
        else:
            self.ema = False
            self.vq_layer = VQLayer(num_embeddings, embedding_dim)


    def forward(self, x):
        x = self.transf_enc.forward(x)

        if self.ema is not True:
            z, quantization_loss, commitment_loss, perplexity = self.vq_layer(x)
            x = z
            x = self.transf_dec.forward(x)
            return x, quantization_loss, commitment_loss, perplexity

        else:
            z, commitment_loss, perplexity = self.vq_layer(x)
            x = z
            x = self.transf_dec.forward(x)
            return x, commitment_loss, perplexity

