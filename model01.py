from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import os
import torch

class VGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # assert input_channel in [1, 3]
        # use pretrained weight of vgg16
        # self.vgg16 = models.vgg16(pretrained=True)
        # vgg16 weights -> is trained from imagenet-1k  
        # features.0.weight False
        # features.0.bias False
        # features.2.weight False
        # features.2.bias False
        # features.5.weight False
        # features.5.bias False
        # features.7.weight False
        # features.7.bias False
        # features.10.weight False
        # features.10.bias False
        # features.12.weight False
        # features.12.bias False
        # features.14.weight False
        # features.14.bias False
        # features.17.weight False
        # features.17.bias False
        # features.19.weight False
        # features.19.bias False
        # features.21.weight False
        # features.21.bias False
        # features.24.weight False
        # features.24.bias False
        # features.26.weight False
        # features.26.bias False
        # features.28.weight False
        # features.28.bias False
        # classifier.0.weight True
        # classifier.0.bias True
        # classifier.3.weight True
        # classifier.3.bias True
        # classifier.6.weight True
        # classifier.6.bias True

        model = models.vgg16(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        for name, param in model.named_parameters():
            param.requires_grad = False

        self.vgg16 = model # last layer shape -> 512 x 7 x 7
        self.fc1 = nn.Linear(25088, 2048)
        self.fc2 = nn.Linear(2048, 11)
        self.relu = nn.ReLU()

        for name, param in model.named_parameters():
            print(name, param.requires_grad)
        for name, param in self.fc1.named_parameters():
            print(name, param.requires_grad)
        for name, param in self.fc2.named_parameters():
            print(name, param.requires_grad)

    def forward(self, x):
        x = self.vgg16(x)
        # print('af vgg', x.shape)
        x = x.view(x.size(0), -1)
        # print('flatten', x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.shape, 'x shape')
        return x


def test_forword():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, 'device')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model = VGG16()
    model = model.to(device)
    inp_size = 360
    # input_tensor = torch.rand(5, 3, 64, 64).to(device)
    input_tensor = torch.rand(1, 3, inp_size, inp_size).to(device)
    print(input_tensor.shape, 'input tensor')
    output = model(input_tensor)
    if type(output) == tuple:
        for out in output:
            print('out shape',out.shape)
    else:
        print('out sh',output.shape)
        print('output',output)
    d = {}
    argmax = torch.argmax(output, 1)
    print(argmax.tolist(), '---')
    a_json = argmax.tolist()[0]
    d['d1'] = str(a_json)
    import json
    with open('test_save.json', 'w') as f:
        json.dump(d, f)
    with open('test_save.json', 'r') as f:
        dat = json.load(f)
        print('---', dat)
        
def test_with_loader():
    from data01 import DME_1k
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # force cpu
    device = torch.device('cpu')
    print(device, 'device')

    dataset = DME_1k('validation', test_mode=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = VGG16()
    model = model.to(device)
    '''
    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'img': self.img[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans
    '''
    for i, dat in enumerate(dataloader):
        img = dat['img'].to(device)
        print(img.shape, 'inp shape from loader')
        output = model(img)
        print(output.shape, 'out shape from model')
        break

    
        
if __name__ == '__main__':
    # test_forword()
    test_with_loader()
        

