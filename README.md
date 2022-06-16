# SS-MLP

# SS-MLP: A Novel Spectral-Spatial MLP Architecture for Hyperspectral Image Classification

PyTorch implementation of SS-MLP: A Novel Spectral-Spatial MLP Architecture for Hyperspectral Image Classification.

![image](fig/dynamic_kernel_generation.png)

# Basic Usage

```
model = SSMLP(num_classes=16, channels=200, patchsize=11)
model.eval()
print(model)
input = torch.randn(100, 200, 11, 11)
y = model(input)
print(y.size())
```

# Paper

[SS-MLP: A Novel Spectral-Spatial MLP Architecture for Hyperspectral Image Classification](https://www.mdpi.com/2072-4292/13/20/4060)

Please cite our paper if you find it useful for your research.

```
@article{meng2021ss,
  title={SS-MLP: A novel spectral-spatial MLP architecture for hyperspectral image classification},
  author={Meng, Zhe and Zhao, Feng and Liang, Miaomiao},
  journal={Remote Sensing},
  volume={13},
  number={20},
  pages={4060},
  year={2021},
  publisher={MDPI}
}
```

# Acknowledgment

This code is partly borrowed from [Involution](https://github.com/d-li14/involution)
