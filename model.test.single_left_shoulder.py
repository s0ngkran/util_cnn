from model import Model
from config import Const
from data01 import MyDataset
from torch.utils.data import DataLoader

def test_basic_forward(model, dataset, img_size, device):
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
        # del img
        # torch.cuda.empty_cache()
        loss = model.cal_loss(pred, keypoint, device)
        if log:
            print('loss', loss)
        if (i > 2):
            break
    print("✅ test_basic_forward")

def test_all():
    kw = {"raw_config": {"data": Const.mode_single_point_left_shoulder}}
    device = "cpu"
    sigma_points = [11.6, 11.6, 11.6]
    sigma_links = [11.6, 11.6, 11.6]
    img_size = 128

    links = MyDataset.get_link()
    model = Model(sigma_points, sigma_links, links, img_size=img_size, **kw).to(device)
    dataset = MyDataset("va", img_size, test_mode=True, **kw)
    test_basic_forward(model, dataset, img_size, device)

if __name__ == "__main__":
    test_all()
