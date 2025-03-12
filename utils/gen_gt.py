import torch
from torchvision.transforms.functional import to_pil_image
import time
from config import Const
import random

try:
    import matplotlib.pyplot as plt
except:
    pass


class GTGen:
    def __init__(self, img_size, sigma_points, sigma_links, links, n_keypoint=19, **kw):
        self.kw = kw
        self.raw_config = kw.get("raw_config")
        self.img_size = img_size
        self.links = links
        self.n_keypoint = n_keypoint
        self.sigma_points = sigma_points
        self.sigma_links = sigma_links
        self.is_custom_mode = kw.get("is_custom_mode", False)
        self.is_use_old_mode = kw.get("is_use_old_mode", False)
        self.is_no_links_custom_mode = kw.get("is_no_links_custom_mode", False)
        self.is_single_point_left_shoulder = (
            self.raw_config.get("data") == Const.mode_single_point_left_shoulder
        )
        self.is_unique_filter_mode = (
            self.raw_config.get("data_in") == "img+unique_filter"
        )

        if not self.is_custom_mode and not self.is_use_old_mode:
            self.is_use_old_mode = True
        if self.is_custom_mode and self.is_use_old_mode:
            raise ValueError("can't use custom mode and old mode at the same time")

        # print('-----')

        if not self.is_use_old_mode:
            if not self.is_no_links_custom_mode:
                for x in sigma_links:
                    assert 0 <= x <= 1, f"sigma link should in range [0, 1] {x=}"
            # RECHECK
            # for x in sigma_points:
            #     assert x > 1, f"sigma point should more than 1; {x=}"

        if self.is_custom_mode:
            if not self.is_no_links_custom_mode:
                assert len(sigma_links) == len(links), (
                    f"{len(sigma_links)} {len(links)}"
                )

            assert len(sigma_points) == self.n_keypoint, (
                f"{len(sigma_points)=} {n_keypoint=}"
            )

        gaussian_size = img_size * 2
        big_gaussians = {}
        if self.is_use_old_mode:
            for sigma_point in sigma_points:
                if str(sigma_point) in big_gaussians:
                    continue
                m = self._gen_gaussian_map(gaussian_size, gaussian_size, sigma_point)
                big_gaussians[str(sigma_point)] = m
            self.big_gaussians = big_gaussians
        else:
            for sigma_point in sigma_points:
                # assert (
                #     0 <= sigma_point <= 1
                # ), f"sigma point should in range [0, 1] but {sigma_point}"
                if str(sigma_point) in big_gaussians:
                    continue
                m = self._gen_gaussian_map(
                    gaussian_size, gaussian_size, sigma_point * self.img_size
                )
                big_gaussians[str(sigma_point)] = m
            self.big_gaussians = big_gaussians

    def __call__(self, keypoints, **args):
        keypoints = self._handle_keypoint_batch(keypoints)
        batch = []

        current_data_augs = args.get("current_data_augs", [])
        if len(current_data_augs) > 0:
            assert len(current_data_augs) == len(keypoints), (
                f"{len(current_data_augs)} {len(keypoints)}"
            )

        for i, keypoint in enumerate(keypoints):
            ar = {}
            if self.is_unique_filter_mode:
                data_aug = current_data_augs[i]
                sp = data_aug.sigma_size
                # index = data_aug.index
                # print(index, 'index')
                ar = {"sp": sp}
            gt_list = self._gen_one_img(keypoint, **ar)
            batch.append(gt_list)
        return batch

    def time(self, keypoints, **kw):
        t1 = time.time()
        keypoints = self._handle_keypoint_batch(keypoints)
        t2 = time.time()
        batch = []
        for keypoint in keypoints:
            assert len(keypoint) == self.n_keypoint, f"{len(keypoint)}"
            gt_list = self._gen_one_img(keypoint)
            batch.append(gt_list)
        t3 = time.time()
        print(t2 - t1, "handle keypoint")
        print(t3 - t2, "gen gt")
        return batch

    def _handle_keypoint_batch(self, keypoints):
        n_batch = len(keypoints[0][0])
        n_keypoint = len(keypoints)
        # print('keypoint1')
        # print(len(keypoints), len(keypoints[0]), len(keypoints[0][0]))
        # 19 2 5
        kps = torch.zeros(n_batch, n_keypoint, 2)
        for i, k in enumerate(keypoints):
            x, y = k
            for b in range(n_batch):
                kps[b, i, 0] = x[b]
                kps[b, i, 1] = y[b]
        return kps

    def _gen_one_img(self, keypoint, **args):  # t0 = time.time()
        size = self.img_size
        x_list = [k[0] * size for k in keypoint]
        y_list = [k[1] * size for k in keypoint]
        xy = x_list, y_list
        gt_list = []
        self.sigma_links = (
            self.sigma_links
            if self.sigma_links is not None
            else [None for i in self.sigma_points]
        )
        # print(keypoint)
        # print('sp', self.sigma_points)
        # print('sl', self.sigma_links)
        # 1/0
        for i, (sp, sl) in enumerate(zip(self.sigma_points, self.sigma_links)):
            if args.get("sp"):
                sp = args.get("sp")
            gt = self._gen_one_size(sp, xy, sl)
            # shape (n_kp, 720, 720)
            gt_list.append(gt)
        # t1 = time.time()
        # t = t1-t0
        # print(t*19*1000,'ttt')
        # 1190/0
        return gt_list

    def _gen_one_size(self, sigma_point, xy, sigma_link):
        x_list, y_list = xy
        gts = self._gen_gts(x_list, y_list, sigma_point)
        gtl = None if sigma_link is None else self._gen_gtl(x_list, y_list, sigma_link)
        return (gts, gtl)

    def _gen_gaussian_map(self, width, height, sigma):
        x = torch.linspace(-width / 2, width / 2, width)
        y = torch.linspace(-height / 2, height / 2, height)
        # already add indexing but the WARNING still remains
        try:
            xv, yv = torch.meshgrid(x, y, indexing="ij")
        except:  # noqa: E722
            xv, yv = torch.meshgrid(x, y)

        gaussian_map = torch.exp(-(xv**2 + yv**2) / (sigma**2))
        # print(gaussian_map.shape) # == (width, hight)
        return gaussian_map

    def _gen_gts(self, x_list, y_list, sigma_point):
        big_gaussian_map = None
        if self.is_use_old_mode:
            big_gaussian_map = self.big_gaussians[str(sigma_point)]
        if self.is_custom_mode:
            assert len(x_list) == len(self.sigma_points)

        size = self.img_size
        tensor_gaussian_map = torch.zeros((len(x_list), size, size))
        for i in range(len(x_list)):
            if self.is_custom_mode:
                sigma_point = self.sigma_points[i]
                big_gaussian_map = self.big_gaussians[str(sigma_point)]
            # crop gaussian map by centering on keypoint
            xi, yi = int(x_list[i]), int(y_list[i])
            # print(xi, yi)
            gaus = big_gaussian_map[
                size - yi : size * 2 - yi, size - xi : size * 2 - xi
            ]
            # if this line error, please check the keypoint is in [0, 1]?
            tensor_gaussian_map[i] = gaus
        return tensor_gaussian_map

    def _gen_gtl(self, x_list, y_list, sigma_link):
        if self.is_custom_mode:
            sigma_links = self.sigma_links
        links = self.links
        if self.is_single_point_left_shoulder:
            return None
        size = self.img_size
        gt_link = torch.zeros((len(links) * 2, size, size))
        for i, (p1, p2) in enumerate(links):
            sigma_link = (
                sigma_link if not self.is_custom_mode else sigma_links[i] * size
            )
            # generate paf
            p1 = torch.tensor([y_list[p1], x_list[p1], 1])
            p2 = torch.tensor([y_list[p2], x_list[p2], 1])
            paf = self._generate_paf(p1, p2, size, sigma_link)
            gt_link[i * 2] = paf[0]
            gt_link[i * 2 + 1] = paf[1]
        return gt_link

    def _generate_paf(self, p1, p2, size, sigma_link):
        paf = torch.zeros((2, size, size))  # (xy, 720, 720)
        small_number = 1e-6

        if p1[2] > 0 and p2[2] > 0:  # Check visibility flags
            diff = p2[:2] - p1[:2]
            # convert to unit vector
            norm = torch.linalg.norm(diff)
            # print()
            # print(norm, 'norm')
            # print(diff, 'diff')

            # if norm > small_number, then diff is not zero vector
            if norm > small_number:
                # unit vector
                v = diff / norm
                v_perpendicular = torch.tensor([-v[1], v[0]])

                # meshgrid
                arr = torch.arange(size)
                x, y = torch.meshgrid(arr, arr)

                dist_x = x - p1[0]
                dist_y = y - p1[1]

                dist_along = v[0] * dist_x + v[1] * dist_y
                dist_perpendicular = torch.abs(
                    v_perpendicular[0] * dist_x + v_perpendicular[1] * dist_y
                )

                # mask distance
                mask1 = dist_along >= 0
                mask2 = dist_along <= norm
                mask3 = dist_perpendicular <= sigma_link
                mask = mask1 & mask2 & mask3

                # add unit vector to paf_x and paf_y
                paf[0, mask] = v[0]
                paf[1, mask] = v[1]
        # convert to torch
        # paf = torch.tensor(paf)
        paf = paf.clone().detach()
        return paf

    def test_gen_gts(self):
        print("test gen gts")

        sigma = 11.6
        keypoint = [(0.1, 0.3), (0.4, 0.1)]
        size = 720
        x_list = [k[0] * size for k in keypoint]
        y_list = [k[1] * size for k in keypoint]
        gts = self._gen_gts(x_list, y_list, sigma)
        assert gts.shape == torch.Size([len(keypoint), size, size]), f"{gts.shape}"
        plt.imshow(gts[0])
        plt.show()

    def test_gen_gtl(self):
        print("test gen gtl")
        size, sigma = 720, 11.6

        keypoint = [(0.1, 0.3), (0.4, 0.1)]
        x_list = [k[0] * size for k in keypoint]
        y_list = [k[1] * size for k in keypoint]
        gtl = self._gen_gtl(x_list, y_list, sigma)
        assert gtl.shape == torch.Size([len(keypoint), size, size]), f"{gtl.shape}"
        plt.imshow(gtl[0])
        plt.show()

    @staticmethod
    def plot_first_heat(axs, gts, gtl):
        axs[0].imshow(gts[0])
        axs[1].imshow(gtl[0])

    @staticmethod
    def plot_mean_heat(axs, gts, gtl):
        axs[0].imshow(gts.mean(0))
        axs[1].imshow(gtl.mean(0))


def test_mini():
    size = 720
    sigma_points = [11.6]
    links = [(0, 1)]
    g = GTGen(size, sigma_points, links)
    g.test_gen_gtl()
    g.test_gen_gts()
    print("passed gts gtl")


def test_gen_gt():
    size = 64
    keypoint = [[[0.1], [0.3]], [[0.4], [0.1]], [[0.5], [0.6]]]
    # 19 2 5
    link = [(0, 1), [1, 2], [0, 2]]
    # sigma_points = [11.6, 7.6, 5.6]
    # sigma_links = [11.6, 7.6, 5.6]

    sigma_points = [x / 720 for x in [11.6, 70.6, 5.6]]
    sigma_links = [x / 720 for x in [11.6, 7.6, 50.6]]

    kw = {
        "is_use_old_mode": False,
        "is_custom_mode": True,
    }

    gen = GTGen(size, sigma_points, sigma_links, link, n_keypoint=3, **kw)
    gt_list = gen(keypoint)
    print("bef")

    n = len(gt_list)
    fig, axs = plt.subplots(n * 2)
    for i, batch_gt in enumerate(gt_list):
        last_stage = batch_gt[-1]
        gts, gtl = last_stage
        print(gts.shape)
        print(gtl.shape)
        print("--")
        # GTGen.plot_first_heat(axs, gts, gtl)
        GTGen.plot_mean_heat(axs, gts, gtl)
    plt.show()


if __name__ == "__main__":
    # test_mini()
    test_gen_gt()
