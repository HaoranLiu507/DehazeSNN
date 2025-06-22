import torch
import torch.nn as nn
import gc
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .snn_cuda import LIFSpike


# Multi-Layer Perceptron (MLP)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU(0.1)):
        """
                MLP module using 1x1 convolution for feature transformation
                :param in_features: Input channels
                :param hidden_features: Hidden layer channels (default is the same as input)
                :param out_features: Output channels (default is the same as input)
                :param act_layer: Activation function (default: LeakyReLU)
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Leaky Integrate-and-Fire (LIF) Module
class LIFModule(nn.Module):
    def __init__(self, dim, lif_bias=True,
                 lif=4, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=0.25):
        """
                LIF (Leaky Integrate-and-Fire) neuron module based on SNN
                :param dim: Feature dimension
                :param lif_bias: Whether to use bias
                :param lif: LIF neuron time steps
                :param lif_fix_tau: Whether to fix membrane potential time constant
                :param lif_fix_vth: Whether to fix membrane potential threshold
                :param lif_init_tau: Initial membrane potential time constant
                :param lif_init_vth: Initial membrane potential threshold
        """
        super().__init__()
        self.lif_init_tau = lif_init_tau
        self.lif_init_vth = lif_init_vth
        self.dim = dim
        self.lif = lif
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=lif_bias)

        self.actn = nn.LeakyReLU(0.1)

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.norm3 = MyNorm(dim)
        self.norm4 = MyNorm(dim)
        self.norm5 = MyNorm(dim)

        self.lif1 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                             init_tau=lif_init_tau, init_vth=lif_init_vth, dim=2)
        self.lif2 = LIFSpike(lif=lif, fix_tau=lif_fix_tau, fix_vth=lif_fix_vth,
                             init_tau=lif_init_tau, init_vth=lif_init_vth, dim=3)
        self.dw1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=lif_bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
        x = self.dw1(x)
        x = self.norm2(x)
        x = self.actn(x)

        x_lr = self.lif1(x)
        x_td = self.lif2(x)
        x_lr = self.conv2_1(x_lr)
        x_lr = self.norm4(x_lr)
        x_td = self.conv2_2(x_td)
        x_td = self.norm5(x_td)
        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)

        x = x_lr + x_td
        x = self.norm3(x)
        x = self.conv3(x)

        return x
    def extra_repr(self) -> str:
        return f'dim={self.dim}, lif={self.lif}, lif_init_tau={self.lif_init_tau}, lif_init_vth={self.lif_init_vth}'


# LIFBlock
class LIFBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., lif_bias=True, drop_path=0.,
                 act_layer=nn.LeakyReLU(0.1), norm_layer=nn.LayerNorm,
                 lif=4, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=0.25):
        """
                :param dim: Input channel size
                :param mlp_ratio: MLP expansion ratio
                :param drop_path: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.lif = lif
        self.norm1 = norm_layer(dim)
        self.lif_module = LIFModule(dim, lif_bias=lif_bias,
                                    lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                    lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # lif block
        x = self.lif_module(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, lif={self.lif}, mlp_ratio={self.mlp_ratio}"


#BasicLayer
class BasicLayer(nn.Module):
    def __init__(self, dim, depth,
                 mlp_ratio=4., lif_bias=True,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 lif=4, lif_fix_tau=False, lif_fix_vth=False, lif_init_tau=0.25, lif_init_vth=0.25):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            LIFBlock(dim=dim,
                     mlp_ratio=mlp_ratio,
                     lif_bias=lif_bias,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                     lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, kernel_size=None):
        """
               Patch embedding layer for feature extraction
               :param patch_size: Size of each patch
               :param in_chans: Number of input channels
               :param embed_dim: Dimension after embedding
               :param norm_layer: Normalization layer
        """
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SKFusion(nn.Module):
    """
        SKFusion Module: Selective Kernel Fusion

        This module implements a Selective Kernel Fusion mechanism, which adaptively selects
        and fuses multi-scale feature representations. It employs an attention-based selection
        mechanism to enhance the most informative feature representations.
    """
    def __init__(self, dim, height=2, reduction=8):
        """
            dim (int): Number of input channels.
            height (int, optional): Number of feature branches. Default is 2.
            reduction (int, optional): Reduction ratio for attention computation. Default is 8.
        """
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


# Normalization layer
def MyNorm(dim):
    return nn.GroupNorm(24, dim)


# Patch Unembedding
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


# DehazeSNN (Spiking Neural Network for Image Dehazing)
class DehazeSNN(nn.Module):
    def __init__(self, in_chans=3,
                 embed_dims=[24, 48, 96, 48, 24], depths=[16, 16, 16, 8, 8],
                 mlp_ratio=[2., 4., 4., 2., 2.], lif_bias=True, drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True,
                 lif=4, lif_fix_tau=False, lif_fix_vth=False,
                 lif_init_tau=0.25, lif_init_vth=0.25):
        super().__init__()
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.patch_size = 4

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=self.embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None, kernel_size=3)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layer1 = BasicLayer(dim=self.embed_dims[0],
                                 depth=depths[0],
                                 mlp_ratio=self.mlp_ratio[0],
                                 lif_bias=lif_bias,
                                 drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                                 norm_layer=norm_layer,
                                 lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                 lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=self.embed_dims[0], embed_dim=self.embed_dims[1])

        self.skip1 = nn.Conv2d(self.embed_dims[0], self.embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=self.embed_dims[1],
                                 depth=depths[1],
                                 mlp_ratio=self.mlp_ratio[1],
                                 lif_bias=lif_bias,
                                 drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                                 norm_layer=norm_layer,
                                 lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                 lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=self.embed_dims[1], embed_dim=self.embed_dims[2])

        self.skip2 = nn.Conv2d(self.embed_dims[1], self.embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=self.embed_dims[2],
                                 depth=depths[2],
                                 mlp_ratio=self.mlp_ratio[2],
                                 lif_bias=lif_bias,
                                 drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                                 norm_layer=norm_layer,
                                 lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                 lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=self.embed_dims[3], embed_dim=self.embed_dims[2])

        self.fusion1 = SKFusion(self.embed_dims[3])

        self.layer4 = BasicLayer(dim=self.embed_dims[3],
                                 depth=depths[3],
                                 mlp_ratio=self.mlp_ratio[3],
                                 lif_bias=lif_bias,
                                 drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                                 norm_layer=norm_layer,
                                 lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                 lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=self.embed_dims[4], embed_dim=self.embed_dims[3])

        self.fusion2 = SKFusion(self.embed_dims[4])

        self.layer5 = BasicLayer(dim=self.embed_dims[4],
                                 depth=depths[4],
                                 mlp_ratio=self.mlp_ratio[4],
                                 lif_bias=lif_bias,
                                 drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                                 norm_layer=norm_layer,
                                 lif=lif, lif_fix_tau=lif_fix_tau, lif_fix_vth=lif_fix_vth,
                                 lif_init_tau=lif_init_tau, lif_init_vth=lif_init_vth)

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=3, embed_dim=self.embed_dims[4], kernel_size=3)

        self.apply(self._init_weights)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)

        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x

        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x

        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x

    def forward(self, x):

        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.forward_features(x)
        x = x[:, :, :H, :W]

        return x
