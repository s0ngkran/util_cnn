from model import Model
from config import Const
from data01 import MyDataset
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from utils.gen_gt import *


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


def test_donut_forward(model, dataset, img_size, device):
    log = True
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i, dat in enumerate(dataloader):
        img = dat["inp"].to(device)
        keypoint = dat["keypoint"]
        if log:
            print(img.shape, "---inp shape from loader")
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
            1/0
        loss = model.cal_loss(pred, keypoint, device, **{
            'plot_donut_mask': plot_donut_mask
        })
        if log:
            print("loss", loss)
        if i > 2:
            break
    print("✅ test_donut_forward")

def config_heat():
    kw = {
        "raw_config": {"data": Const.mode_single_point_left_shoulder, "mode": "heatmap"}
    }
    device = "cpu"
    sigma_points = [11.6, 11.6, 11.6]
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
    sigma_points = [11.6, 11.6, 11.6]
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
    test_donut_forward(*out)


if __name__ == "__main__":
    test_all()
