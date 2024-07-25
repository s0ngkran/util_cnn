import torch.nn as nn
import torchvision.models as models
import os
import torch

class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        print('Model kwargs:', kwargs)
        if kwargs.get('train_all') == True:
            print('train all layers of vgg')
            model = models.vgg16()
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            print('load pretrained weight...')
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
            for name, param in model.named_parameters():
                param.requires_grad = False

        self.vgg16 = model # last layer shape -> 512 x 7 x 7

        n_hidden1 = 2048
        n_hidden2 = 2048
        n_out = 11 if kwargs.get('out11') == True else 30
        if kwargs.get('no_bn_dr') == True:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, n_hidden1),
                # nn.BatchNorm1d(n_hidden1),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(n_hidden1, n_hidden2),
                # nn.BatchNorm1d(n_hidden2),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(n_hidden2, n_out),
                nn.Sigmoid()
            )
            return
        elif kwargs.get('no_drop') == True:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, n_hidden1),
                nn.BatchNorm1d(n_hidden1),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(n_hidden1, n_hidden2),
                nn.BatchNorm1d(n_hidden2),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(n_hidden2, n_out),
                nn.Sigmoid()
            )
            return
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hidden2, n_out),
            nn.Sigmoid()
        )
        


    def forward(self, x):
        x = self.vgg16(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test_forword(device):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model = Model().to(device)
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
    from data01 import MyDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # force cpu
    # device = torch.device('cpu')
    print(device, 'device')

    dataset = MyDataset('tr', test_mode=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = Model()
    model = model.to(device)
    for i, dat in enumerate(dataloader):
        img = dat['inp'].to(device)
        print(img.shape, 'inp shape from loader')
        output = model(img)
        print(output.shape, 'out shape from model')
        break
        
if __name__ == '__main__':
    test_forword('cpu')
    # test_with_loader()
    # Model(**{
    #     'no_drop': True
    # })
        
