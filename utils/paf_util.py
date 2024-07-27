import torch.nn as nn
import math
import torchvision.models as models

def make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True):
    layers = []
    layers += [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
    if use_bn:
        layers += [nn.BatchNorm2d(feat_out, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, stage1):
        super(Stage, self).__init__()
        inp_feats = backend_outp_feats
        if stage1:
            self.block1 = self.make_paf_block_stage1(inp_feats, n_joints)
            self.block2 = self.make_paf_block_stage1(inp_feats, n_paf)
        else:
            inp_feats = backend_outp_feats + n_joints + n_paf
            self.block1 = self.make_paf_block_stage2(inp_feats, n_joints)
            self.block2 = self.make_paf_block_stage2(inp_feats, n_paf)
        init(self.block1)
        init(self.block2)


    def make_paf_block_stage1(self, inp_feats, output_feats):
        layers = [make_standard_block(inp_feats, 128, 3),
                  make_standard_block(128, 128, 3),
                  make_standard_block(128, 128, 3),
                  make_standard_block(128, 512, 1, 1, 0)]
        layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
        return nn.Sequential(*layers)


    def make_paf_block_stage2(self, inp_feats, output_feats):
        layers = [make_standard_block(inp_feats, 128, 7, 1, 3),
                  make_standard_block(128, 128, 7, 1, 3),
                  make_standard_block(128, 128, 7, 1, 3),
                  make_standard_block(128, 128, 7, 1, 3),
                  make_standard_block(128, 128, 7, 1, 3),
                  make_standard_block(128, 128, 1, 1, 0)]
        layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
        return nn.Sequential(*layers)


    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        return y1, y2


class VGG19(nn.Module):
    def __init__(self, use_bn=True, no_weight=False): 
        # original no bn
        super().__init__()
        if use_bn:
            # vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            vgg = models.vgg19(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:23]
        else:
            vgg = models.vgg19_bn(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:33]
        self.vgg = nn.Sequential(*layers_to_use)
        self.feature_extractor = nn.Sequential(make_standard_block(512, 256, 3),
                                               make_standard_block(256, 128, 3))
        init(self.feature_extractor)

    def forward(self, x):
        x = self.vgg(x)
        x = self.feature_extractor(x)
        # print('out vgg', x.shape)
        return x
