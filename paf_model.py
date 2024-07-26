from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import os
import torch
import time
import torch.nn.functional as F
from utils.paf_util import *
from utils.gen_gt import *


class PAF(nn.Module):
    def __init__(self, n_point=19, n_link=18, n_stage=3, img_size=720):
        super().__init__()
        assert (n_stage > 0)
        n_paf = n_link * 2
        self.img_size = img_size
        self.n_joint = n_point
        self.n_link = n_link
        self.n_paf = n_paf
        self.n_stage = n_stage
        self.backend = VGG19() 
        backend_outp_feats=128
        stages = [Stage(backend_outp_feats, n_point, n_paf, True)]
        for i in range(n_stage - 1):
            stages.append(Stage(backend_outp_feats, n_point, n_paf, False))
        self.stages = nn.ModuleList(stages)
        self.gt_gen = None

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
    
    def cal_loss(self, pred, gt):
        heatmap_outs, paf_outs = pred
        # scale up each out from 90 to 720 
        loss_point = 0
        loss_link = 0
        n_stage = len(heatmap_outs)
        n_batch = len(heatmap_outs[0])
        size = (self.img_size, self.img_size)
        for i in range(n_stage):
            # scale to 720 (original size)
            # t1 = time.time()
            heatmaps = F.interpolate(heatmap_outs[i], size=size, mode='bilinear').cpu()
            pafs = F.interpolate(paf_outs[i], size=size, mode='bilinear').cpu()
            # t2 = time.time()
            batch_gts = []
            batch_gtl = []
            for b in range(n_batch):
                gts = gt[b][i][0]
                gtl = gt[b][i][1]
                batch_gts.append(gts)
                batch_gtl.append(gtl)
            batch_gts = torch.stack(batch_gts)
            batch_gtl = torch.stack(batch_gtl)
            # t3 = time.time()
            loss_point += F.mse_loss(heatmaps, batch_gts)
            loss_link += F.mse_loss(pafs, batch_gtl)
            # t4 = time.time()
        # print(t2-t1, 'interpolate')
        # print(t3-t2, 'prepare gt')
        # print(t4-t3, 'cal loss')
        sum_loss = loss_point + loss_link
        return sum_loss

    def init_gt_generator(self, img_size, sigma_points, links):
        assert len(sigma_points) == self.n_stage
        assert len(links) == self.n_link
        self.gt_gen = GTGen(img_size, sigma_points, links)
    def gen_gt(self, keypoint, sigma_points, sigma_links):
        assert len(sigma_points) == self.n_stage
        assert len(sigma_links) == self.n_stage
        gt = self.gt_gen(keypoint, sigma_points, sigma_links)
        return gt


def test_forword(device='cuda'):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model = PAF().to(device)
    img_size = 128
    img_size = 720
    input_tensor = torch.rand(2, 3, img_size, img_size).to(device)
    print(input_tensor.shape, 'input tensor')
    output = model(input_tensor)
    if type(output) == tuple:
        for out in output:
            print()
            print('out shape',len(out))
            for o in out:
                print(o.shape)

def test_loss(device='cuda'):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model = PAF().to(device)
    img_size = 128
    img_size = 720

    k = [(.2, .3) for i in range(19)]
    keypoints = [k, k]
    n_batch = len(keypoints)
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    model.init_gt_generator(img_size, sigma_points, links)
    input_tensor = torch.rand(n_batch, 3, img_size, img_size).to(device)
    print(input_tensor.shape, 'input tensor')
    t1 = time.time()
    pred = model(input_tensor)
    t2 = time.time()
    gt = model.gen_gt(keypoints, sigma_points, sigma_links)
    t3 = time.time()
    # assert n_batch == len(gt)
    loss = model.cal_loss(pred, gt)
    t4 = time.time()

    print(t2-t1, 'pred')
    print(t3-t2, 'gen gt')
    print(t4-t3, 'cal loss')
    print('loss', loss)
    # 6.959 pred
    # 0.142 gen gt
    # 0.530 cal loss
        # 0.107 interpolate
        # 0.022 prepare gt
        # 0.047 cal loss
    # loss tensor(41.8182, grad_fn=<AddBackward0>)
        
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
    # test_forword('cpu')
    # test_with_loader()
    test_loss('cpu')
        
