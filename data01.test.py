from data01 import MyDataset
from config import Const

try:
    import matplotlib.pyplot as plt
except:
    pass


def stop():
    1 / 0

def plot_img(img, keypoint=None):
    img_color_mean = img.permute(1, 2, 0).numpy()
    plt.imshow(img_color_mean)

    img_size = img.shape[-1]

    if keypoint is not None:
        for i, (x, y) in enumerate(keypoint):
            x, y = x * img_size, y * img_size
            plt.plot(x, y, "or")
            plt.text(x, y, str(i))
    plt.show()

def plot():
    img_size = 64
    kw = {
        "raw_config": {"data": Const.mode_single_point_left_shoulder}
    }
    data = MyDataset("va", img_size, **kw)
    for i, d in enumerate(data):
        if i < 1:
            continue
        img = d["inp"]
        keypoint = d["keypoint"]
        gt = d["gt"]
        # print(keypoint)
        print(i, keypoint)
        # plot_img(img, keypoint)
        print(type(gt))
        stop()

        for x, y in keypoint:
            if x > 1:
                print(x)
                plot_img(img, keypoint)
                1 / 0

            if y > 1:
                print(y)
                1 / 0


if __name__ == "__main__":
    # test()
    plot()
    # set seed of random number generator
    # random.seed(0)
    # torch.manual_seed(0)

    # test()
    # plot()
