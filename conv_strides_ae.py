import torch.nn as nn
import torch.nn.functional as F



class CommonLayers(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.norm = nn.BatchNorm2d(n_features)

    def forward(self, x):
        return F.relu(self.norm(x))



class BlockConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size // 2)
        )
        self.common = CommonLayers(out_ch)

    def forward(self, x):
        return self.common(self.conv2d(x))



class BlockDeconv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__()
        output_padding = 0
        if stride == 2:
            output_padding = 1
        if stride == 4:
            output_padding = 3
        self.deconv2d = nn.ConvTranspose2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size // 2), output_padding=output_padding
        )
        self.common = CommonLayers(out_ch)

    def forward(self, x):
        return self.common(self.deconv2d(x))



class ConvNet(nn.Module):

    def __init__(self, latent_dim=10, filter1=8, filter2=16, filter3=32,
                 enc_kernel_ls=[3, 3, 3],
                 enc_strides=[2, 2, 2], overall_dim=1024):
        super().__init__()
        layer_filters_in = [2, filter1, filter2]
        layer_filters_out = [filter1, filter2, filter3]
        deconv_filters_in = [filter3, filter3, filter2, filter1]
        overall_scale_factor = int(overall_dim / ((enc_strides[0] * enc_strides[1] * enc_strides[2]) ** 2))
        self.convs = nn.ModuleList(
            [
                BlockConv2d(filters_in, filters_out, kernel, strides) \
                for filters_in, filters_out, kernel, strides \
                in zip(layer_filters_in, layer_filters_out, enc_kernel_ls, enc_strides)
            ]
        )
        self.linear_enc = nn.Linear(in_features=filter3 * overall_scale_factor, out_features=latent_dim)
        self.linear_dec = nn.Linear(in_features=latent_dim, out_features=filter3 * overall_scale_factor)
        deconv_filters_out = layer_filters_out.copy()
        deconv_filters_out.reverse()
        dec_kernel_ls = enc_kernel_ls.copy()
        dec_kernel_ls.reverse()
        dec_strides = enc_strides.copy()
        dec_strides.reverse()
        self.deconvs = nn.ModuleList(
            [
                BlockDeconv2d(filters_in, filters_out, kernel, strides) \
                for filters_in, filters_out, kernel, strides in
                zip(deconv_filters_in, deconv_filters_out, dec_kernel_ls, dec_strides)
            ]
        )
        self.last_deconv = nn.ConvTranspose2d(
            in_channels=deconv_filters_out[-1], out_channels=2, kernel_size=3, padding=1
        )

    def forward(self, x):
        for i, con in enumerate(self.convs):
            x = self.convs[i](x)
        shape_x = x.shape
        x = nn.Flatten()(x)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.view(shape_x)
        for j, decon in enumerate(self.deconvs):
            x = self.deconvs[j](x)
        return self.last_deconv(x)