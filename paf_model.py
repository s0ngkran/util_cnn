from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import os
import torch
import math
import time
import torch.nn.functional as F

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

class PAF(nn.Module):
    def __init__(self, backend=None, backend_outp_feats=128, n_joints=19, n_paf=36, n_stages=3):
        super().__init__()
        assert (n_stages > 0)
        backend = VGG19() if backend is None else backend
        self.backend = backend
        stages = [Stage(backend_outp_feats, n_joints, n_paf, True)]
        for i in range(n_stages - 1):
            stages.append(Stage(backend_outp_feats, n_joints, n_paf, False))
        self.stages = nn.ModuleList(stages)
        self.init_gt_generator()

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs = []
        paf_outs = []
        for i, stage in enumerate(self.stages):
            heatmap_out, paf_out = stage(cur_feats)
            heatmap_outs.append(heatmap_out)
            paf_outs.append(paf_out)
            cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
        return heatmap_outs, paf_outs
    
    def loss_func(self, pred, gt):
        heatmap_outs, paf_outs = pred
        gts, gtl = gt
        
        # scale up each out from 90 to 720 
        loss_point = 0
        loss_link = 0
        for i in range(3):
            # scale to 720
            heatmaps = F.interpolate(heatmap_outs[i], size=(720, 720), mode='bilinear').cpu()
            loss_point += F.mse_loss(heatmaps, gts)

            pafs = F.interpolate(paf_outs[i], size=(720, 720), mode='bilinear').cpu()
            loss_link += F.mse_loss(pafs, gtl)
        return loss_point + loss_link

    def init_gt_generator(self):
        width, height, sigma = 720, 720, 11.6
        self.gaussian = self._gen_gaussian_map(width, height, sigma)
        
    def _gen_gaussian_map(self, width, height, sigma):
        x = torch.linspace(-width / 2, width / 2, width)
        y = torch.linspace(-height / 2, height / 2, height)
        xv, yv = torch.meshgrid(x, y, indexing='xy')
        gaussian_map = torch.exp(-(xv ** 2 + yv ** 2) / (sigma ** 2))
        # print(gaussian_map.shape) # == (width, hight)
        return gaussian_map

    def gen_gt(self, keypoint, link):
        w, h = 720, 720
        x_list = [k[0]*w for k in keypoint]
        y_list = [k[1]*h for k in keypoint]
        gts = self._gen_gts(x_list, y_list, self.gaussian_map)
        gtl = self._gen_gtl(x_list, y_list, link)
        return (gts, gtl)

    def _gen_gts(self, x_list, y_list, big_gaussian_map):
        w, h = 720, 720
        tensor_gaussian_map = torch.zeros((len(x_list), h, w))
        for i in range(len(x_list)):
            # crop gaussian map by centering on keypoint
            xi, yi = int(x_list[i]), int(y_list[i])
            gaus = big_gaussian_map[h-yi:h*2 - yi, w-xi:w*2 - xi]
            tensor_gaussian_map[i] = gaus
        return tensor_gaussian_map

    def _gen_gtl(self, x_list, y_list, links):
        size = 720
        gt_link = torch.zeros((len(links) * 2, size, size))
        for j, link in enumerate(links):
            # generate paf
            p1 = np.array([x_list[link[0]], y_list[link[0]], 1])
            p2 = np.array([x_list[link[1]], y_list[link[1]], 1])
            paf = self._generate_paf(p1, p2, size, sigma_link)
            gt_link[j * 2] = paf[0]
            gt_link[j * 2 + 1] = paf[1]
        return gt_link

    def _generate_paf(self, p1, p2, size, sigma_link):
        paf = np.zeros((2, size, size)) # (xy, 720, 720)
        
        if p1[2] > 0 and p2[2] > 0:  # Check visibility flags
            diff = p2[:2] - p1[:2]
            # convert to unit vector
            norm = np.linalg.norm(diff)
            # print()
            # print(norm, 'norm')
            # print(diff, 'diff')
            
            # if norm > 1e-6, then diff is not zero vector
            if norm > 1e-6:
                # unit vector
                v = diff / norm
                v_perpendicular = np.array([-v[1], v[0]])

                # meshgrid
                x, y = np.meshgrid(np.arange(size), np.arange(size))

                dist_x = x - p1[0]
                dist_y = y - p1[1]

                dist_along = v[0] *  dist_x + v[1] * dist_y
                dist_perpendicular = np.abs(v_perpendicular[0] * dist_x + v_perpendicular[1] * dist_y)
                
                # mask distance
                mask1 = dist_along >= 0
                mask2 = dist_along <= norm
                mask3 = dist_perpendicular <= sigma_link
                mask = mask1 & mask2 & mask3

                # add unit vector to paf_x and paf_y
                paf[0, mask] = v[0]
                paf[1, mask] = v[1]
        # convert to torch
        paf = torch.tensor(paf)
        return paf 

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
    def __init__(self, use_bn=True): 
        # original no bn
        super().__init__()
        if use_bn:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
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

def test_forword(device):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # model = VGG19()
    model = PAF().to(device)
    inp_size = 128
    inp_size = 720
    # input_tensor = torch.rand(5, 3, 64, 64).to(device)
    input_tensor = torch.rand(1, 3, inp_size, inp_size).to(device)
    print(input_tensor.shape, 'input tensor')
    output = model(input_tensor)
    if type(output) == tuple:
        for out in output:
            print()
            print('out shape',len(out))
            for o in out:
                print(o.shape)
        
def test_with_loader():
    from data01 import Dataset_S1_1000_PAF
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, 'device')

    dataset = Dataset_S1_1000_PAF('validation', test_mode=True)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=8)
    model = PAF().to(device)
    '''
    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'img': self.img[idx],
            'ground_truth': self.ground_truth[idx],
            'gt_link': self.gt_link[idx],
            'gt_point': gt_point,
        }
        return ans
    '''
    for i, dat in enumerate(dataloader):
        img = dat['img'].to(device)
        print(img.shape, 'inp shape from loader')
        pred = model(img)
        print(len(pred), 'out len from model')
        # print gt_link and gt_point
        gt = dat['gt']
        print('gt', gt[0].shape, gt[1].shape)

        # heatmap_outs, paf_outs = output
        # print each out shape
        for i, heatmap_out in enumerate(pred[0]):
            print(heatmap_out.shape, 'heatmap_out shape')
        
        for i, paf_out in enumerate(pred[1]):
            print(paf_out.shape, 'paf_out shape')

        # cal loss
        loss = model.loss_func(pred, dat)
        print(loss, 'loss')
        break


# torch.Size([5, 36, 720, 720]) gt_link shape
# torch.Size([5, 19, 720, 720]) gt_point shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
        
if __name__ == '__main__':
    test_forword('cpu')
    # test_with_loader()
        
