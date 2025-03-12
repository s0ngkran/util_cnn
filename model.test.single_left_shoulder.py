from model import Model
from config import Const
from data01 import MyDataset
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from utils.gen_gt import *
import random
import time


def test_heatmap_forward(model, dataset, img_size, device):
    log = True

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i, dat in enumerate(dataloader):
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        if log:
            print(img.shape, "---inp shape from loader")

        pred = model(img)
        if log:
            print(pred[0][-1].shape, "---pred shape from model")
        loss = model.cal_loss(pred, keypoint, device)
        if log:
            print("loss", loss)
        if i > 2:
            break
    print("✅ test_heatmap_forward")


def test_forward_with_plot(name, model, dataset, img_size, device):
    log = True
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for iteration, dat in enumerate(dataloader):
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        if log:
            print(img.shape, "---inp shape from loader" + str(iteration))
            # plt.imshow(img[0].permute(1, 2, 0).numpy())
            # plt.show()

        pred = model(img)
        if log:
            print(pred[0][-1].shape, "---pred shape from model")

        def plot_donut_mask(donut_mask):
            return
            # merge img and donut_mask
            img_color_mean = img[0].permute(1, 2, 0).numpy()
            plt.imshow(img_color_mean)
            plt.imshow(donut_mask[0][0], alpha=0.5)
            plt.show()
            1 / 0

        def plot_multi_sigma_size(batch_gts):
            return
            # merge img and donut_mask
            img_color_mean = img[0].permute(1, 2, 0).numpy()
            plt.imshow(img_color_mean)
            # plt.imshow(batch_gts[0][0], alpha=0.7)

            hilight = batch_gts[0][0] > 0.5
            plt.imshow(hilight, alpha=0.7)
            plt.show()
            1 / 0

        loss = model.cal_loss(
            pred,
            keypoint,
            device,
            **{
                "plot_donut_mask": plot_donut_mask, # delete return to enable
                "plot_multi_sigma_size": plot_multi_sigma_size, # delete return to enable
            },
        )

        if log:
            print("loss", loss)
        if iteration > 1:
            break
    print("✅ test_" + name)


def config_label_c():
    ref = 11.6
    ref1 = ref * 1
    ref2 = ref * 2
    ref4 = ref * 4
    kw = {
        "raw_config": {
            "data": Const.mode_single_point_left_shoulder,
            "img_size": 128,
            "mode": "label-encoding-add-channel",
            "data_aug": [ref1, ref2, ref4],
            "data_aug_weight": [1, 2, 1],
            "data_in": "img+channel",
            "loss": "mse"
        }
    }
    device = "cpu"
    sigma_points = [ref1, ref2, ref4]
    sigma_links = [11.6, 11.6, 11.6]
    img_size = 128

    links = MyDataset.get_link()
    model = Model(sigma_points, sigma_links, links, img_size=img_size, **kw).to(device)
    dataset = MyDataset("va", img_size, test_mode=True, **kw)
    return model, dataset, img_size, device

def config_label_f():
    ref = 11.6
    ref1 = ref * 1
    ref2 = ref * 2
    ref4 = ref * 4
    kw = {
        "raw_config": {
            "data": Const.mode_single_point_left_shoulder,
            "img_size": 128,
            "mode": "label-encoding-filter",
            "data_aug": [ref1, ref2, ref4],
            "data_aug_weight": [1, 2, 1],
            "data_in": "img+unique_filter",
            "loss": "mse",
        }
    }
    device = "cpu"
    sigma_points = [ref1, ref2, ref4]
    sigma_links = [11.6, 11.6, 11.6]
    img_size = 128

    links = MyDataset.get_link()
    model = Model(sigma_points, sigma_links, links, img_size=img_size, **kw).to(device)
    dataset = MyDataset("va", img_size, test_mode=True, **kw)
    return model, dataset, img_size, device

def config_heat(kw=None):
    if kw is None:
        kw = {
            "raw_config": {
                "data": Const.mode_single_point_left_shoulder,
                "mode": "heatmap",
            }
        }
    device = "cpu"
    ref = 11.6
    ref2 = ref * 2
    sigma_points = [ref2, ref2, ref2]
    sigma_links = [11.6, 11.6, 11.6]
    img_size = 128

    links = MyDataset.get_link()
    model = Model(sigma_points, sigma_links, links, img_size=img_size, **kw).to(device)
    dataset = MyDataset("va", img_size, test_mode=True, **kw)
    return model, dataset, img_size, device


def config_donut():
    kw = {
        "raw_config": {"data": Const.mode_single_point_left_shoulder, "mode": "donut"}
    }
    device = "cpu"
    ref = 11.6
    ref2 = ref * 2
    sigma_points = [ref2, ref2, ref2]
    sigma_links = [11.6, 11.6, 11.6]
    img_size = 128

    links = MyDataset.get_link()
    model = Model(sigma_points, sigma_links, links, img_size=img_size, **kw).to(device)
    dataset = MyDataset("va", img_size, test_mode=True, **kw)
    return model, dataset, img_size, device


def test_all():
    out = config_heat()
    test_heatmap_forward(*out)
    out = config_donut()
    test_forward_with_plot("donut", *out)

    # this failed VGG weight only -> 3 channels
    # out = config_label_c()
    # test_forward_with_plot("s-label-c", *out)

    out = config_label_f()
    test_forward_with_plot("s-label-f", *out)

if __name__ == "__main__":
    test_all()
