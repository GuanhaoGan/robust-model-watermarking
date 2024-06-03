import torch.nn as nn
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs) * self.mask

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    base = nn.Conv2d(10,10,3)
    masks = torch.ones(10)
    masks[:7]=0
    masks = masks[None, :, None, None]
    model = MaskedLayer(base, masks)
    inputs = torch.randn(128,10,5,5)
    outputs = base(inputs) * masks
    print(outputs.norm(p=1, dim=[0,2,3]))

