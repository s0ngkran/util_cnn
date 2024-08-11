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
from torch.utils.data import DataLoader
from data01 import MyDataset, Data
try:
    import matplotlib.pyplot as plt
except:
    pass


class PAF(nn.Module):
    def __init__(self, sigma_points, sigma_links, links, n_point=19, n_link=18, n_stage=3, img_size=720, no_weight=False, **kw):
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
        self.backend = VGG19(no_weight=no_weight) 
        backend_outp_feats=128
        stages = [Stage(backend_outp_feats, n_point, n_paf, True)]
        for i in range(n_stage - 1):
            stages.append(Stage(backend_outp_feats, n_point, n_paf, False))
        self.stages = nn.ModuleList(stages)
        self.gt_gen = self._init_gt_generator(img_size, sigma_points, sigma_links, links)

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

    def _init_gt_generator(self, img_size, sigma_points, sigma_links, links):
        assert len(sigma_points) == self.n_stage
        assert len(links) == self.n_link
        gt_gen = GTGen(img_size, sigma_points, sigma_links, links)
        return gt_gen
        
    def gen_gt(self, keypoint):
        gt = self.gt_gen(keypoint)
        return gt

    def get_pred(self, output, func):
        keypoint_batch = self.get_keypoints(output)
        pred_batch = [func(k) for k in keypoint_batch]
        return pred_batch

    def get_keypoints(self, output, from_gt=False):
        if from_gt:
            heat_batch = self._handle_gt_batch(output)
        else:
            heat_batch = self._handle_output(output)
        keypoints = self._get_keypoints_from_batch(heat_batch)
        return keypoints

    def _handle_output(self, output):
        heat, paf = output
        heat_batch = heat[-1]
        return heat_batch

    def _handle_gt_batch(self, gt_batch):
        # do not commit this code
        heat, paf = last_stage
        heat = heat[-1]
        heat_batch = []
        for gt in gt_batch:
            last_stage = gt[-1]
            heat, paf = last_stage
            heat_batch.append(heat)
        heat_batch = torch.stack(heat_batch)
        return heat_batch

    def _get_keypoints_from_batch(self, heat_batch):
        batch_size, num_maps, height, width = heat_batch.shape
        keypoints = []
        for i in range(batch_size):
            kps = []
            for j in range(num_maps):
                heatmap = heat_batch[i, j, :, :]
                max_index = torch.argmax(heatmap).item()
                y, x = divmod(max_index, width)
                kps.append((x, y))
            keypoints.append(kps)
        return keypoints





def test_forword(device='cuda'):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    img_size = 720
    model = Model(sigma_points, sigma_links, links, img_size=img_size, no_weight=True).to(device)
    # img_size = 720
    input_tensor = torch.rand(2, 3, img_size, img_size).to(device)
    print(input_tensor.shape, 'input tensor')
    output = model(input_tensor)
    if type(output) == tuple:
        for out in output:
            print()
            print('out shape',len(out))
            for o in out:
                print(o.shape)

def plot_img_keypoint(img, keypoint):
    print(keypoint)
    # plt.imshow(img)

def test_convert_heat(device='cpu', dataset='va', img_size=128):
    my_data = MyDataset(dataset,  img_size, no_aug=True, test_mode=False)
    loader = DataLoader(my_data, batch_size=10, shuffle=False, num_workers=10, drop_last=False)
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = my_data.get_link()
    model = PAF(
        sigma_points,
        sigma_links,
        links,
        img_size=img_size,
        no_weight=True
    ).to(device)
    n = len(loader)
    cnt = 0
    fail = []
    for d in loader:
        cnt += 1
        print(cnt, n)
        imgs = d['inp']
        keypoint_batch = d['keypoint']
        keys = d['key']

        data = [my_data.get_data(k) for k in keys]

        gt_batch = model.gen_gt(keypoint_batch)
        new_keypoint_batch = model.get_keypoints(gt_batch)

        # print('keypoint batch',new_keypoint_batch)
        for i in range(len(imgs)):
            # img = imgs[i]
            dat = data[i]
            keypoint = new_keypoint_batch[i]
            pred = Data.pred_from_keypoint(keypoint)
            if pred != dat.gt:
                out= f'{pred}, {dat.gt}'
                print((dat.key, out))
                fail.append((dat.key, out))


            # plt.imshow(torch.mean(img, dim=0))
            # # plt.imshow(torch.mean(heat_batch[i], dim=0))
            # for i, (x,y) in enumerate(keypoint):
            #     plt.plot(x,y,'ro')
            #     plt.text(x+10,y,str(i))
                
            # plt.title(str(i))
            # plt.show()
    print('passed', img_size, dataset, len(fail))
    print(len(fail), fail)

    # va 21 9
    # va 64 9
    # va 128 7
    # va 256 7
    # va 360 0
    # va 720 0
    return len(fail)




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
    img_size = 64
    model = Model(
        [10,10,10],
        [10,10,10],
        MyDataset.get_link(),
        img_size=img_size,
    ).to(device)
    dataset = MyDataset('va', img_size, test_mode=True)
    dataloader = DataLoader(dataset,  batch_size=5, shuffle=True)
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
        print(img.shape, 'inp shape from loader')

        pred = model(img)
        print(pred[0][-1].shape)
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
        super().__init__(sigma_points, sigma_links, links, n_point=n_point, n_link=n_link, n_stage=n_stage, img_size=img_size, **kw)


# torch.Size([5, 36, 720, 720]) gt_link shape
# torch.Size([5, 19, 720, 720]) gt_point shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape

# torch.Size([5, 19, 90, 90]) -> keypoint -> tfs

     
if __name__ == '__main__':
    # test_forword('cuda')
    # test_loss('cuda')
    test_with_loader('cuda')
    # for dataset in ['te', 'va', 'tr']:
    #     for img_size in [32, 64, 128, 256, 360, 720]:
    #         test_convert_heat('cpu', dataset, img_size)
        
