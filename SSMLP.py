'''
SS-MLP

Meng Z, Zhao F, Liang M. SS-MLP: A Novel 
Spectral-Spatial MLP Architecture for Hyperspectral 
Image Classification[J]. Remote Sensing, 2021, 13(20): 4060.

'''
import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, int(dim * expansion_factor)),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(int(dim * expansion_factor), dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, dim, depth, patch_size=1, expansion_factor=4, dropout=0.5):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    expansion_factor1 = 0.5
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor1, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
    )


class SSMLP(nn.Module):
    def __init__(self, num_classes, channels, patchsize=11, dim=24, depth=3):
        super(SSMLP, self).__init__()

        self.model = MLPMixer(image_size=patchsize,
                              channels=channels,
                              dim=dim,
                              depth=depth,
                              )
        self.classifi = nn.Sequential(nn.LayerNorm(dim),
                                      Reduce('b n c -> b c', 'mean'),
                                      nn.Linear(dim, num_classes))

    def forward(self, x):
        out = self.model(x)
        out = self.classifi(out)
        return out


if __name__ == '__main__':

    model = SSMLP(num_classes=16, channels=200, patchsize=11)
    model.eval()
    print(model)
    input = torch.randn(100, 200, 11, 11)
    y = model(input)
    print(y.size())