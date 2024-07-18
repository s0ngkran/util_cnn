from torchvision import transforms
import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
class MyLinear(nn.Module):
    def __init__(self, n_inp, n_out, act_func=nn.ReLU) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(n_inp, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.act = act_func()
    def forward(self, x):
        x = self.linear_layer(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ANNPipe(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        assert mode in ['abs', 'rel']
        act = nn.ReLU
        if mode == 'rel':
            self.layer1 = MyLinear(21*3-3, 60, act)
        else:
            self.layer1 = MyLinear(21*3, 60, act)
        self.layer2 = MyLinear(60, 40, act)
        self.layer3 = MyLinear(40, 30, act)
        self.layer4 = MyLinear(30, 30, nn.Sigmoid)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def test_forword():
    model = ANNPipe('rel')
    input_tensor = torch.rand(2, 21*3-3)
    output = model(input_tensor)
    print(output.shape)
    assert output.shape == torch.Size([2,30])

    model = ANNPipe('abs')
    input_tensor = torch.rand(4, 21*3)
    output = model(input_tensor)
    print(output.shape)
    assert output.shape == torch.Size([4,30])

if __name__ =='__main__':
    test_forword()