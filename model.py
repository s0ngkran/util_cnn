from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import os
import torch
import time
from torchvision.transforms.functional import to_pil_image 
import torch.nn.functional as F
from utils.paf_util import *
from utils.gen_gt import *
from utils.cuda import *


class PAF(nn.Module):
    def __init__(self, sigma_points, sigma_links, links, n_point=19, n_link=18, n_stage=3, img_size=720, **kw):
        super().__init__()
        self.sigma_points = sigma_points
        self.sigma_links = sigma_links
        self.links = links
        n_paf = n_link * 2
        self.img_size = img_size
        self.n_joint = n_point
        self.n_link = n_link
        self.n_paf = n_paf
        self.n_stage = n_stage
        assert n_link == len(links)
        assert n_stage > 0
        assert len(sigma_points) == n_stage
        assert len(sigma_links) == n_stage
        self.backend = VGG19() 
        backend_outp_feats=128
        stages = [Stage(backend_outp_feats, n_point, n_paf, True)]
        for i in range(n_stage - 1):
            stages.append(Stage(backend_outp_feats, n_point, n_paf, False))
        self.stages = nn.ModuleList(stages)
        self.gt_gen = self.init_gt_generator(img_size, sigma_points, sigma_links, links)

    def __str__(self):
        txt = ['PAF model']
        txt.append(f'n_joint={self.n_joint}')
        txt.append(f'n_link={self.n_link}')
        txt.append(f'n_stage={self.n_stage}')
        txt.append(f'sig_point={self.sigma_points}')
        txt.append(f'sig_link={self.sigma_links}')
        txt = ' '.join(txt)
        return txt

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
    
    def cal_loss(self, pred, gt, device='cuda'):
        gt = self.gen_gt(gt)
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
            heatmaps = F.interpolate(heatmap_outs[i], size=size, mode='bilinear').to(device)
            pafs = F.interpolate(paf_outs[i], size=size, mode='bilinear').to(device)
            # t2 = time.time()
            batch_gts = []
            batch_gtl = []
            for b in range(n_batch):
                gts = gt[b][i][0]
                gtl = gt[b][i][1]
                batch_gts.append(gts)
                batch_gtl.append(gtl)
            batch_gts = torch.stack(batch_gts).to(device)
            batch_gtl = torch.stack(batch_gtl).to(device)

            # print(heatmaps.shape, 'heat')
            # print(pafs.shape, 'paf')
            # print(batch_gts.shape, 'gts')
            # print(batch_gtl.shape, 'gtl')
            # t3 = time.time()
            loss_point += F.mse_loss(heatmaps, batch_gts)
            loss_link += F.mse_loss(pafs, batch_gtl)
            # t4 = time.time()
        # print(t2-t1, 'interpolate')
        # print(t3-t2, 'prepare gt')
        # print(t4-t3, 'cal loss')
        sum_loss = loss_point + loss_link
        return sum_loss

    def init_gt_generator(self, img_size, sigma_points, sigma_links, links):
        assert len(sigma_points) == self.n_stage
        assert len(links) == self.n_link
        gt_gen = GTGen(img_size, sigma_points, sigma_links, links)
        return gt_gen
        
    def gen_gt(self, keypoint):
        gt = self.gt_gen(keypoint)
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
    img_size = 128
    img_size = 720

    k = [(.2, .3) for i in range(19)]
    keypoints = [k, k]
    n_batch = len(keypoints)
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    model = PAF(sigma_points, sigma_links, links).to(device)
    input_tensor = torch.rand(n_batch, 3, img_size, img_size).to(device)
    print(input_tensor.shape, 'input tensor')
    pred = model(input_tensor)
    loss = model.cal_loss(pred, keypoints, 'cpu')
    print('loss', loss)
    # 6.959 pred
    # 0.142 gen gt
    # 0.530 cal loss
        # 0.107 interpolate
        # 0.022 prepare gt
        # 0.047 cal loss
    # loss tensor(41.8182, grad_fn=<AddBackward0>)
        
def test_with_loader(device='cuda'):
    from data01 import MyDataset
    from torch.utils.data import DataLoader

    img_size = 720
    model = PAF(
        [10,10,10],
        [10,10,10],
        MyDataset.get_link(),
        img_size=img_size,
    ).to(device)
    dataset = MyDataset('va', img_size, test_mode=True)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    '''
    def __getitem__(self, idx):
        ans = {
            'inp': img,
            'keypoint': data.keypoint,
            'poh_gt': data.gt,
            'raw': data,
        }
        return ans
    '''
    t = []
    m = []
    for i, dat in enumerate(dataloader):
        t0 = time.time()
        img = dat['inp'].to(device)
        keypoint = dat['keypoint']
        # print(img.shape, 'inp shape from loader')

        pred = model(img)
        # del img
        # torch.cuda.empty_cache()
        t1 = time.time()
        device = 'cuda'
        loss = model.cal_loss(pred, keypoint, device)
        t2 = time.time()
        loss.backward()
        t3 = time.time()
        # print(loss, 'loss', )
        # print(t2-t1, 'time loss', device)
        # print(t3-t2, 'time backward', device)
        t.append(t3-t0)
        # mem = get_gpu_memory_info()
        # m.append(mem)
        if i > 2:
            break
    print(sum(t), 'sum', device)
    print(m)

    def save_img(gt):
        gt = loss
        gts, gtl = gt[0][0]
        gts = torch.mean(gts, dim=0)*10
        gtl = torch.mean(gtl, dim=0)*10 +.6
        print('gts',gts.shape, torch.min(gts), torch.max(gts))
        print('gtl',gtl.shape, torch.min(gtl), torch.max(gtl))
        img = img[0]
        img = torch.mean(img, dim=0)*.2 +.5
        print('img',img.shape, torch.min(img), torch.max(img))

        x = to_pil_image(img)
        x.save('temp1.jpg')
        x = to_pil_image(gts)
        x.save('temp2.jpg')
        x = to_pil_image(gtl)
        x.save('temp3.jpg')
        x = to_pil_image(img*.2+gts*.8)
        x.save('temp4.jpg')
    # save_img(gt)

    print()
    print('passed')

class Model(PAF):
    def __init__(self, sigma_points, sigma_links, links, n_point=19, n_link=18, n_stage=3, img_size=720, **kw):
        super().__init__(sigma_points, sigma_links, links, n_point=19, n_link=18, n_stage=3, img_size=720, **kw)


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
    # test_loss('cuda')
    test_with_loader('cuda')
        
