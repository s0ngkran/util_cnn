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
from config import config, Const
import random
from utils.pattern import *
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
except:
    pass


@dataclass
class DataAug:
    index: str
    sigma_size: str
    unique_filter: str


class PAF(nn.Module):
    def __init__(
        self,
        sigma_points,
        sigma_links,
        links,
        n_point=19,
        n_link=18,
        n_stage=3,
        img_size=720,
        no_weight=False,
        bi_mode=False,
        bi_thres=0,
        **kw,
    ):
        super().__init__()
        self.kw = kw
        self.raw_config = kw.get("raw_config", {})
        if bi_mode:
            assert n_stage == 3
        self.bi_mode = bi_mode
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
        self.is_custom_mode = kw.get("is_custom_mode", False)
        self.is_no_links_mode = kw.get("is_no_links_mode", False)
        self.is_no_links_custom_mode = kw.get("is_no_links_custom_mode", False)
        if self.is_no_links_mode:  # config startswith n
            self.n_paf = 0
            assert n_stage == 3
            assert len(sigma_points) == n_stage
            if self.sigma_links == [0]:
                self.sigma_links = None
            assert self.sigma_links is None, f"{self.sigma_links=}"
        elif self.is_no_links_custom_mode:  # config startswith o
            self.n_paf = 0
            assert self.sigma_links is None
        elif not self.is_custom_mode:
            assert len(sigma_links) == n_stage
            assert len(sigma_points) == n_stage, f"{len(sigma_points)=} {n_stage=}"
            print(kw)

        self.is_single_point_left_shoulder = (
            self.raw_config.get("data") == Const.mode_single_point_left_shoulder
        )
        if self.is_single_point_left_shoulder:
            self.n_joint = 1
            self.n_paf = 0
        self.is_donut_mode = self.raw_config.get("mode") == "donut"

        self.backend = VGG19(no_weight=no_weight)
        backend_outp_feats = 128
        stages = [Stage(backend_outp_feats, self.n_joint, self.n_paf, True)]
        for i in range(n_stage - 1):
            stages.append(Stage(backend_outp_feats, self.n_joint, self.n_paf, False))
        self.stages = nn.ModuleList(stages)

        self.gt_gen = self._init_gt_generator(
            img_size, sigma_points, sigma_links, links, **kw
        )
        self.bce = None
        self.bi_thres = bi_thres
        if bi_mode:
            assert bi_thres > 0
            self.bce = nn.BCEWithLogitsLoss()
            print()
            print("message from model: bi_mode activated")
        self.current_data_augs = []
        self.data_in = self.raw_config.get("data_in")
        self.data_aug = self.raw_config.get("data_aug")
        self.data_aug_weight = self.raw_config.get("data_aug_weight")
        # make data_aug btw -1 and 1
        if len(self.data_aug) > 1:
            self.scaled_data_aug = self.get_scaled_data_aug(self.data_aug)
            self.unique_filters = []
            assert len(self.data_aug) == 3, "only 3 patterns are prepared"
            for i, sigma in enumerate(self.data_aug):  # no need to scale
                unique_filter = generate_unique_filter_linear(i, self.img_size)  # noqa: F405
                self.unique_filters.append(DataAug(i, sigma, unique_filter))
                # print(unique_filter.shape, 'unique filter shape--')
                # plot
            #     plt.imshow(unique_filter)
            #     plt.title(f'unique filter sigma={i}')
            #     plt.show()
            # 1/0

        # print(self.scaled_data_aug, 'scaled data aug') # -1, -0.3, 1

    def get_scaled_data_aug(self, data_aug):
        mx = max(data_aug)
        mn = min(data_aug)
        return [(x - mn) / (mx - mn) * 2 - 1 for x in data_aug]

    def __str__(self):
        txt = ["PAF model"]
        txt.append(f"n_joint={self.n_joint}")
        txt.append(f"n_link={self.n_link}")
        txt.append(f"n_stage={self.n_stage}")
        txt.append(f"sig_point={self.sigma_points}")
        txt.append(f"sig_link={self.sigma_links}")
        txt = " ".join(txt)
        return txt

    def add_channel(self, batch_img):
        # add the 4th channel
        n = len(batch_img)
        self.randomed_sigmas = random.choices(
            self.scaled_data_aug, weights=self.data_aug_weight, k=n
        )
        # print size before adding channel
        # print(batch_img[0].shape, 'before adding channel')
        sigma_maps = torch.stack(
            [torch.ones_like(batch_img[0, 0]) * sigma for sigma in self.randomed_sigmas]
        )
        sigma_maps = sigma_maps.unsqueeze(1)
        new_batch = torch.cat([batch_img, sigma_maps], dim=1)
        # print(new_batch[0].shape, 'after adding channel') # should be 4 instead of 3
        # new_batch shape = 5, 4, 128, 128

        # loop plt plot each image
        # for c in range(4):
        # print first 3 values of each channel
        # print(new_batch[0][c][0][0:10], 'c=',c)
        # break
        # plt.imshow(new_batch[i][0].cpu())
        # plt.title(f'img {i}')
        # plt.show()
        # 1/0
        return new_batch

    def add_unique_filter(self, batch_img):
        n = len(batch_img)
        # add these for cal_loss step
        self.current_data_augs = random.choices(
            self.unique_filters, weights=self.data_aug_weight, k=n
        )
        # for d in self.current_data_augs:
        #     print(d.index, d.sigma_size)
        
        imgs = []
        # apply unique filter to each image
        for i, (unique_filter, img) in enumerate(zip(self.current_data_augs, batch_img)):
            unique_filter = unique_filter.unique_filter
            
            prepared_unique_filter = torch.stack([unique_filter for _ in range(3)]) 
            # print(prepared_unique_filter.shape, 'prepared unique filter shape')
            
            # apply to img
            device = img.device
            prepared_unique_filter = prepared_unique_filter.to(device)

            img = img * prepared_unique_filter
            imgs.append(img)
            # print(img.shape, 'img shape')
            
            #plot
            # for i in range(3):
            #     plt.imshow(img[i].cpu())
            #     plt.title(f'img {i}')
            #     plt.show()
            # 1/0
        new_batch = torch.stack(imgs)
        return new_batch

    def forward(self, batch_img):
        if self.is_single_point_left_shoulder:
            if self.data_in == "img+channel":
                assert False, "pretrained VGG19 use 3 channels"
                batch_img = self.add_channel(batch_img)
            if self.data_in == "img+unique_filter":
                batch_img = self.add_unique_filter(batch_img)

        return self.forward_normal(batch_img)

    def forward_normal(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs = []
        paf_outs = []
        for i, stage in enumerate(self.stages):
            heatmap_out, paf_out = stage(cur_feats)
            heatmap_outs.append(heatmap_out)
            if paf_out is not None:
                paf_outs.append(paf_out)
                cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
            else:
                cur_feats = torch.cat([img_feats, heatmap_out], 1)
        return heatmap_outs, paf_outs

    def to_binary(self, tensor, threshold):
        return (tensor >= threshold).float()

    @staticmethod
    def reshape_gt(n_batch, gt, i, device):
        batch_gts = []
        batch_gtl = []
        # gts = gt[0][i][0]
        # mx = torch.max(a)
        # mn = torch.min(a)
        # print(f'{gts.shape=}') # 19, 360, 360
        # print(torch.max(gts))
        # 1/0
        for b in range(n_batch):
            gts = gt[b][i][0]
            gtl = gt[b][i][1]
            batch_gts.append(gts)
            batch_gtl.append(gtl)
        batch_gts = torch.stack(batch_gts).to(device)
        batch_gtl = None if batch_gtl[0] is None else torch.stack(batch_gtl).to(device)
        return batch_gts, batch_gtl

    def cal_donut_loss(
        self, heatmaps, batch_gts, donut_thres_out, donut_thres_in, **kw
    ):
        donut_mask_out = (batch_gts > donut_thres_out).float()
        # _1111111_
        """
        -------
        -11111-
        -11111-
        -11111-
        -------
        """
        donut_mask_in = (batch_gts < donut_thres_in).float()
        # 1111___1111
        """
        1111111
        1111111
        11---11
        1111111
        1111111
        """
        donut_mask = donut_mask_out * donut_mask_in
        # __11___11___
        """
        -------
        -11111-
        -1---1-
        -11111-
        -------
        """
        inverted_donut_mask = 1 - donut_mask  ## checked
        masked_heatmaps = heatmaps * inverted_donut_mask  ## checked
        masked_batch_gts = batch_gts * inverted_donut_mask  ## checked
        loss = F.mse_loss(masked_heatmaps, masked_batch_gts)

        # plt.imshow(donut_mask_out[0][0].cpu())
        # plt.title('donut mask out')
        # plt.show()
        # plt.imshow(donut_mask_in[0][0].cpu())
        # plt.title('donut mask in')
        # plt.show()

        # plt.imshow(donut_mask[0][0].cpu())
        # plt.title('donut mask')
        # plt.show()

        plot = kw.get("plot_donut_mask")
        if plot:
            plot(donut_mask)

        # batch_gts
        # plt.imshow(batch_gts[0][0].cpu())
        # plt.title('batch gts')
        # plt.show()

        # plt.imshow(masked_heatmaps[0][0].detach().numpy())
        # plt.title('masked heatmaps')
        # plt.show()

        # plt.imshow(masked_batch_gts[0][0].detach().numpy())
        # plt.title('masked batch gts')
        # plt.show()

        # print(loss, '--los')
        return loss

    def cal_loss_point(self, batch_gts, heatmap_out_i, i, size, device, **kw):
        heatmaps = F.interpolate(heatmap_out_i, size=size, mode="bilinear").to(device)
        # handle bi_mode
        loss_point = 0
        if self.is_donut_mode:
            # hard code thres value
            # donut_thres_out = 0.15 # 1x 128
            # donut_thres_in = 0.6 # 1x 128
            donut_thres_out = 0.5  # 2x 128
            donut_thres_in = 0.85  # 2x 128
            loss_point = self.cal_donut_loss(
                heatmaps, batch_gts, donut_thres_out, donut_thres_in, **kw
            )
        elif self.bi_mode and i <= 1:
            # already assert n_stages == 3 for bi_mode
            batch_gts = self.to_binary(batch_gts, self.bi_thres)
            loss_point = self.bce(heatmaps, batch_gts)
        else:
            func = kw.get("plot_multi_sigma_size")
            if func:
                func(batch_gts)
            loss_point = F.mse_loss(heatmaps, batch_gts)
        return loss_point

    def cal_loss_link(self, batch_gtl, paf_outs, i, size, device):
        pafs = F.interpolate(paf_outs[i], size=size, mode="bilinear").to(device)
        loss_link = F.mse_loss(pafs, batch_gtl)
        return loss_link

    def cal_loss(self, pred, gt, device="cuda", **kw):
        args = {}
        if self.is_single_point_left_shoulder and len(self.current_data_augs) > 0:
            # send unique filters aug data to gt_gen
            args = {
                'current_data_augs': self.current_data_augs,
            }
        gt = self.gt_gen(gt, **args)
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
            # t2 = time.time()
            batch_gts, batch_gtl = self.reshape_gt(n_batch, gt, i, device)
            # print(batch_gts.shape, 'batch gts shape')

            # print(heatmaps.shape, 'heat')
            # print(pafs.shape, 'paf')
            # print(batch_gts.shape, 'gts')
            # print(batch_gtl.shape, 'gtl')
            # t3 = time.time()
            loss_point += self.cal_loss_point(
                batch_gts, heatmap_outs[i], i, size, device, **kw
            )
            if len(paf_outs) > 0:
                loss_link += self.cal_loss_link(batch_gtl, paf_outs, i, size, device)

            # save img of each batch_gts
            # def save_example():
            #     for i, (gts, gtl) in enumerate(zip(batch_gts, batch_gtl)):
            #         x = to_pil_image(gts[0]) # only first keypoint
            #         x.save(f'temp_gts_{i}.jpg')
            #         x = to_pil_image(gtl[0])
            #         x.save(f'temp_gtl_{i}.jpg')
            #     1/0

            # print('p stage',i, heatmaps.shape, batch_gts.shape)
            # print('l stage',i, pafs.shape, batch_gtl.shape)
            # t4 = time.time()
        # print(t2-t1, 'interpolate')
        # print(t3-t2, 'prepare gt')
        # print(t4-t3, 'cal loss')
        sum_loss = loss_point + loss_link
        return sum_loss

    def _init_gt_generator(self, img_size, sigma_points, sigma_links, links, **kw):
        if not self.is_custom_mode and not self.is_no_links_custom_mode:
            assert len(sigma_points) == self.n_stage
        assert len(links) == self.n_link
        gt_gen = GTGen(img_size, sigma_points, sigma_links, links, **kw)
        return gt_gen

    def gen_gt(self, keypoint):
        gt = self.gt_gen(keypoint)
        return gt

    def get_pred(self, output, func):
        keypoint_batch = self.get_keypoints(output)
        pred_batch = [func(k) for k in keypoint_batch]
        # each k has 19 keypoints
        return pred_batch

    # this function for single point
    def get_pred_for_dist_err(self, output):
        keypoint_batch = self.get_keypoints(output)
        # each k has only on (x, y)
        return keypoint_batch


    @staticmethod
    def gt_batch_to_list(gt_keypoints):
        result = [
            [] for _ in range(len(gt_keypoints[0][0]))
        ]  # Initialize list of 5 elements (for 5 batches)

        for keypoints in gt_keypoints:  # Iterate over the 19 keypoints
            x_positions, y_positions = keypoints  # Unpack the x and y positions

            for i in range(len(x_positions)):  # For each batch (5 in this case)
                x = x_positions[i].item()  # Get the x value for the batch
                y = y_positions[i].item()  # Get the y value for the batch
                result[i].append(
                    (y, x)
                )  # Append the (x, y) tuple for the current keypoint

        return result

    @staticmethod
    def get_keypoint_from_a_heatmap(heatmap, original_size):
        """
        |0|1|2|
        0 -> 0.5/3 = 0+0.5/3
        1 -> 1.5/3 = 1+0.5/3
        2 -> 2.5/3 = 2+0.5/3
        """
        keypoint = (heatmap == torch.max(heatmap)).nonzero()[0]
        keypoint = torch.div(keypoint + 0.5, original_size)
        return keypoint.tolist()

    def get_keypoint_batch_by_scale_up(self, output, device="cuda"):
        heatmap_outs, paf_outs = output
        # scale up each out from 90 to 720
        original_size = (self.img_size, self.img_size)

        last_heatmaps = heatmap_outs[-1]
        n_batch = len(last_heatmaps)
        # scale to 720 (original size)
        heatmaps_original_size = F.interpolate(
            last_heatmaps, size=original_size, mode="bilinear"
        ).to(device)

        keypoint_batch = []
        for batch in range(n_batch):
            heatmaps = heatmaps_original_size[batch]
            keypoints = [
                PAF.get_keypoint_from_a_heatmap(heatmap, original_size[0])
                for heatmap in heatmaps
            ]
            keypoint_batch.append(keypoints)
        return keypoint_batch

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


def test_forword(device="cuda"):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    img_size = 64
    model = Model(
        sigma_points, sigma_links, links, img_size=img_size, no_weight=True
    ).to(device)
    # img_size = 720
    input_tensor = torch.rand(2, 3, img_size, img_size).to(device)
    print(input_tensor.shape, "input tensor")
    output = model(input_tensor)
    if type(output) == tuple:
        for out in output:
            print()
            print("out shape", len(out))
            for o in out:
                print(o.shape)


def plot_img_keypoint(img, keypoint):
    print(keypoint)
    # plt.imshow(img)


def test_convert_heat(device="cpu", dataset="va", img_size=128):
    my_data = MyDataset(dataset, img_size, no_aug=True, test_mode=False)
    loader = DataLoader(
        my_data, batch_size=10, shuffle=False, num_workers=10, drop_last=False
    )
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = my_data.get_link()
    model = PAF(sigma_points, sigma_links, links, img_size=img_size, no_weight=True).to(
        device
    )
    n = len(loader)
    cnt = 0
    fail = []
    for d in loader:
        cnt += 1
        print(cnt, n)
        imgs = d["inp"]
        keypoint_batch = d["keypoint"]
        keys = d["key"]

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
                out = f"{pred}, {dat.gt}"
                print((dat.key, out))
                fail.append((dat.key, out))

            # plt.imshow(torch.mean(img, dim=0))
            # # plt.imshow(torch.mean(heat_batch[i], dim=0))
            # for i, (x,y) in enumerate(keypoint):
            #     plt.plot(x,y,'ro')
            #     plt.text(x+10,y,str(i))

            # plt.title(str(i))
            # plt.show()
    print("passed", img_size, dataset, len(fail))
    print(len(fail), fail)

    # va 21 9
    # va 64 9
    # va 128 7
    # va 256 7
    # va 360 0
    # va 720 0
    return len(fail)


def test_loss(device="cuda"):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    img_size = 128
    img_size = 720

    k = [(0.2, 0.3) for i in range(19)]
    keypoints = [k, k]
    n_batch = len(keypoints)
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    model = PAF(sigma_points, sigma_links, links).to(device)
    input_tensor = torch.rand(n_batch, 3, img_size, img_size).to(device)
    print(input_tensor.shape, "input tensor")
    pred = model(input_tensor)
    loss = model.cal_loss(pred, keypoints, "cpu")
    print("loss", loss)
    # 6.959 pred
    # 0.142 gen gt
    # 0.530 cal loss
    # 0.107 interpolate
    # 0.022 prepare gt
    # 0.047 cal loss
    # loss tensor(41.8182, grad_fn=<AddBackward0>)


def test_with_loader(device="cuda"):
    img_size = 720
    img_size = 64
    model = Model(
        [10, 10, 10],
        [10, 10, 10],
        MyDataset.get_link(),
        img_size=img_size,
    ).to(device)
    dataset = MyDataset("va", img_size, test_mode=True)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    """
    def __getitem__(self, idx):
        ans = {
            'inp': img,
            'keypoint': data.keypoint,
            'poh_gt': data.gt,
            'raw': data,
        }
        return ans
    """
    t = []
    m = []
    for i, dat in enumerate(dataloader):
        t0 = time.time()
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        print(img.shape, "inp shape from loader")

        pred = model(img)
        print(pred[0][-1].shape)
        # del img
        # torch.cuda.empty_cache()
        t1 = time.time()
        device = "cuda"
        loss = model.cal_loss(pred, keypoint, device)
        t2 = time.time()
        loss.backward()
        t3 = time.time()
        # print(loss, 'loss', )
        # print(t2-t1, 'time loss', device)
        # print(t3-t2, 'time backward', device)
        t.append(t3 - t0)
        # mem = get_gpu_memory_info()
        # m.append(mem)
        break
        if i > 2:
            break
    print(sum(t), "sum", device)
    print(m)

    def save_img(gt):
        gt = loss
        gts, gtl = gt[0][0]
        gts = torch.mean(gts, dim=0) * 10
        gtl = torch.mean(gtl, dim=0) * 10 + 0.6
        print("gts", gts.shape, torch.min(gts), torch.max(gts))
        print("gtl", gtl.shape, torch.min(gtl), torch.max(gtl))
        img = img[0]
        img = torch.mean(img, dim=0) * 0.2 + 0.5
        print("img", img.shape, torch.min(img), torch.max(img))

        x = to_pil_image(img)
        x.save("temp1.jpg")
        x = to_pil_image(gts)
        x.save("temp2.jpg")
        x = to_pil_image(gtl)
        x.save("temp3.jpg")
        x = to_pil_image(img * 0.2 + gts * 0.8)
        x.save("temp4.jpg")

    # save_img(gt)

    print()
    print("passed")


class Model(PAF):
    def __init__(
        self,
        sigma_points,
        sigma_links,
        links,
        n_point=19,
        n_link=18,
        n_stage=3,
        img_size=720,
        bi_mode=False,
        bi_thres=0,
        **kw,
    ):
        super().__init__(
            sigma_points,
            sigma_links,
            links,
            n_point=n_point,
            n_link=n_link,
            n_stage=n_stage,
            img_size=img_size,
            bi_mode=bi_mode,
            bi_thres=bi_thres,
            **kw,
        )


# torch.Size([5, 36, 720, 720]) gt_link shape
# torch.Size([5, 19, 720, 720]) gt_point shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 19, 90, 90]) heatmap_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape
# torch.Size([5, 36, 90, 90]) paf_out shape

# torch.Size([5, 19, 90, 90]) -> keypoint -> tfs


def test_bi_model(device):
    k = [(0.2, 0.3) for i in range(19)]
    keypoints = [k, k]
    n_batch = len(keypoints)
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    links = [(0, 2) for i in range(18)]
    model = Model(sigma_points, sigma_links, links).to(device)
    assert model.fake_stages == [False, False, False]

    model = Model(sigma_points, sigma_links, links, bi_mode=True).to(device)
    assert model.fake_stages == [True, True, False]
    print()
    print("passed model bi_mode")


def test_bi_mode_feed(device):
    from data01 import MyDataset
    from torch.utils.data import DataLoader

    bi_mode = True
    bi_thres = 0.7
    img_size = 720
    img_size = 64
    model = Model(
        [10, 10, 10],
        [10, 10, 10],
        MyDataset.get_link(),
        img_size=img_size,
        bi_mode=bi_mode,
        bi_thres=bi_thres,
    ).to(device)
    dataset = MyDataset("va", img_size, test_mode=True)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    """
    def __getitem__(self, idx):
        ans = {
            'inp': img,
            'keypoint': data.keypoint,
            'poh_gt': data.gt,
            'raw': data,
        }
        return ans
    """
    t = []
    m = []
    for i, dat in enumerate(dataloader):
        t0 = time.time()
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        print(img.shape, "inp shape from loader")

        pred = model(img)
        print(pred[0][-1].shape)
        # del img
        # torch.cuda.empty_cache()
        t1 = time.time()
        device = "cuda"
        loss = model.cal_loss(pred, keypoint, device)
        t2 = time.time()
        loss.backward()
        t3 = time.time()
        # print(loss, 'loss', )
        # print(t2-t1, 'time loss', device)
        # print(t3-t2, 'time backward', device)
        t.append(t3 - t0)
        # mem = get_gpu_memory_info()
        # m.append(mem)
        break
        if i > 2:
            break
    print(sum(t), "sum time", device)
    print(m)
    print("loss", loss)


def test_custom_mode_with_loader(device="cuda"):
    img_size = 720
    # img_size = 64

    training = config()["m3"]

    kw = {
        "is_custom_mode": True,
    }
    model = Model(
        training["sigma_points"],
        training["sigma_links"],
        MyDataset.get_link(),
        img_size=img_size,
        **kw,
    ).to(device)
    dataset = MyDataset("va", img_size, test_mode=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    n = 1
    fig, axs = plt.subplots(n * 2)
    for i, dat in enumerate(dataloader):
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        print(img.shape, "inp shape from loader")

        pred = model(img)
        print(pred[0][-1].shape)
        loss = model.cal_loss(pred, keypoint, device)
        print(loss)

        gt = model.gen_gt(keypoint)
        batch, stage = 0, -1
        gts, gtl = gt[batch][stage][0], gt[batch][stage][1]
        batch_gts, batch_gtl = Model.reshape_gt(2, gt, stage, device)
        # print(batch_gts.shape, batch_gtl.shape)

        GTGen.plot_mean_heat(axs, batch_gts[0].to("cpu"), batch_gtl[0].to("cpu"))
        # plt.show()
        plt.savefig(f"{i}.png")
        break


if __name__ == "__main__":
    device = "cuda"
    # test_custom_mode_with_loader(device)
    # test_bi_model('cuda')
    # test_bi_mode_feed(device)
    test_forword("cuda")
    # test_loss('cuda')
    # test_with_loader(device)
    # for dataset in ['te', 'va', 'tr']:
    #     for img_size in [32, 64, 128, 256, 360, 720]:
    #         test_convert_heat('cpu', dataset, img_size)
