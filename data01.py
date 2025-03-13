from torch.utils.data import Dataset
import torch
import os
import time
import json
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from torch.utils.data import DataLoader
from utils.func import get_dist
import random
import math
from config import Const

try:
    import matplotlib.pyplot as plt
except:  # noqa: E722
    pass

root = "../poh_vr_1k"
if not os.path.exists(root):
    root = "../Research/data_zip/poh_vr_1k"


@dataclass
class Data:
    ind: str
    img_path: str
    gt: int
    keypoint: list
    dataset: str
    key: str = ""

    def __post_init__(self):
        self.key = self.img_path.split("/")[-1]
        self.img_path = os.path.join(root, self.img_path)
        assert type(self.gt) is int
        assert self.gt >= 0 and self.gt <= 10
        assert len(self.keypoint) == 19
        assert self.dataset in ["tr", "va", "te"]

    @staticmethod
    def pred_from_keypoint(keypoint):
        assert len(keypoint) == 19
        palm_ind_list = [i for i in range(7, 19)]
        palm_ind_list.remove(17)  # remove pinky
        palm = [keypoint[i] for i in palm_ind_list]
        assert len(palm) == 11
        index_finger_tip = keypoint[0]
        dists = torch.tensor([get_dist(p, index_finger_tip) for p in palm])
        ind = torch.argmin(dists)
        ind = ind.item()
        return ind

    def plot(self, img_size=360, palm=None):
        # print(self.img_path, '--img')
        img = Image.open(self.img_path)
        img = img.resize((img_size, img_size))
        # img =cv2.imread(self.img_path)

        # palm and index finger tip
        if palm is not None:
            kp = self.keypoint
            w = img_size
            palm = [(p[0] * w, p[1] * w) for p in palm]
            for i, (x, y) in enumerate(palm):
                plt.plot(x, y, "or")
                plt.text(x, y, str(i))
            plt.plot(kp[0][0] * img_size, kp[0][1] * img_size, "ob")

        plt.imshow(img)


class MyDataset(Dataset):
    cache_folder = ".cache_image"

    def __init__(self, dataset, img_size=0, test_mode=False, **kwargs):
        assert dataset in ["tr", "va", "te"]
        assert img_size in [64, 128, 256, 360, 720]
        self.raw_config = kwargs.get('raw_config', {})
        self.mode = self.raw_config.get('data')
        self.dataset = dataset
        self.img_size = img_size
        data = self.read_data(dataset)
        self.no_aug = True if kwargs.get("no_aug") is True else False
        if test_mode:
            data = data[:100]
            print(f"checking mode {len(data)=}")
        self.data = data
        self.link = self.get_link()

        if not os.path.exists(self.cache_folder):
            os.mkdir(self.cache_folder)
        self.cache = {}

    def get_data(self, key):
        for d in self.data:
            if key == d.key:
                return d
    
    def manage_keypoint(self, keypoint):
        if self.mode == Const.mode_single_point_left_shoulder:
            keypoint = self.only_single_point_left_shoulder(keypoint)
        return keypoint

    def only_single_point_left_shoulder(self, keypoint): # tested
        index_of_left_shoulder = 4
        keypoint = [k for i, k in enumerate(keypoint) if i == index_of_left_shoulder]
        assert len(keypoint) == 1
        return keypoint

    def __getitem__(self, idx):
        data = self.data[idx]
        img_path = data.img_path
        new_keypoint = self.manage_keypoint(data.keypoint)
        img, keypoint = self.load_img(img_path, new_keypoint)
        ans = {
            "inp": img,
            "keypoint": keypoint,
            "gt": data.gt, # gt index ex. 4(int)
            "key": data.key,
        }
        return ans

    @staticmethod
    def get_link():
        links = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [7, 18],
            [18, 12],
            [17, 16],
            [14, 15],
            [13, 12],
            [10, 11],
            [11, 12],
            [12, 15],
            [15, 16],
        ]
        return links

    def get_cache_path(self, image_path):
        last = image_path.split("/")[-1]
        return os.path.join(self.cache_folder, last)

    def check_bad_keypoint(self, keypoint):
        for x, y in keypoint:
            if x > 1 or y > 1 or x < 0 or y < 0:
                return True
        return False

    def gen_rotated_data(self, img_pil, keypoint, max_angle):
        # tested
        """
        random angle within max_angle
        ex. max_angle = 30, then -30 to 30
        """
        keypoint = keypoint.copy()
        angle = random.uniform(-max_angle, max_angle)
        # rotate keypoint
        keypoint = torch.tensor(keypoint)
        center = torch.tensor([0.5, 0.5])
        angle = torch.tensor([angle]) * math.pi / 180
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rot_matrix = torch.stack([cos, -sin, sin, cos]).view(2, 2)
        keypoint = (keypoint - center) @ rot_matrix + center
        keypoint = keypoint.tolist()
        if self.check_bad_keypoint(keypoint):
            return False
        img = img_pil.rotate(angle)
        return img, keypoint
    
    def gen_cropped_data_for_single_point(self, img_pil, keypoint, img_size):
        keypoint = keypoint.copy()
        assert len(keypoint) == 1
        img_pil = img_pil.resize((img_size, img_size))
        h, w = img_size, img_size
        x, y = keypoint[0]

        pad_x = 0.3
        pad_y = 0.3

        kp_pad_x = x * 0.3
        kp_pad_y = y * 0.3

        left = max(0, x - kp_pad_x)
        right = min(w, x + kp_pad_x)
        top = max(0, y - kp_pad_y)
        bottom = min(h, y + kp_pad_y)

        crop_x1 = max(0, left - pad_x)
        crop_x2 = min(w, right + pad_x)
        crop_y1 = max(0, top - pad_y)
        crop_y2 = min(h, bottom + pad_y)

        cropped_image = img_pil.crop(
            (crop_x1 * img_size, crop_y1 * img_size, crop_x2 * img_size, crop_y2 * img_size)
        )

        new_x = x - crop_x1
        new_y = y - crop_y1
        new_keypoint = (new_x, new_y)

        if self.check_bad_keypoint(keypoint):
            return False

        cropped_image = cropped_image.resize((img_size, img_size))
        return cropped_image, new_keypoint

    def gen_cropped_data(self, img_pil, keypoint, img_size):
        keypoint = keypoint.copy()
        img_pil = img_pil.resize((img_size, img_size))
        x_list = [p[0] for p in keypoint]
        y_list = [p[1] for p in keypoint]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        dist_x, dist_y = x_max - x_min, y_max - y_min
        min_dist = min(dist_x, dist_y)
        padding = min_dist * 0.3  # 30% padding
        # print('x_min, x_max, y_min, y_max =', x_min, x_max, y_min, y_max)
        # print('padding =', padding)

        inner_x_min = max(0, x_min - padding)
        inner_x_max = min(1, x_max + padding)
        inner_y_min = max(0, y_min - padding)
        inner_y_max = min(1, y_max + padding)

        # print('inner_x_min, inner_x_max, inner_y_min, inner_y_max =', inner_x_min, inner_x_max, inner_y_min, inner_y_max)
        # print('img_size =', img_size)

        space_left = [0, inner_x_min]
        space_right = [inner_x_max, 1]
        space_top = [0, inner_y_min]
        space_bottom = [inner_y_max, 1]

        # random crop
        x_min = random.uniform(*space_left)
        x_max = random.uniform(*space_right)
        y_min = random.uniform(*space_top)
        y_max = random.uniform(*space_bottom)

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        min_size = min(crop_w, crop_h)

        if crop_w != min_size:
            x_min = inner_x_min
            x_max = x_min + min_size
        else:
            y_min = inner_y_min
            y_max = y_min + min_size

        # update keypoint
        keypoint = torch.tensor(keypoint)
        keypoint[:, 0] = (keypoint[:, 0] - x_min) / (x_max - x_min)
        keypoint[:, 1] = (keypoint[:, 1] - y_min) / (y_max - y_min)
        keypoint = keypoint.tolist()
        # print(keypoint)

        if self.check_bad_keypoint(keypoint):
            return False

        cropped_image = img_pil.crop(
            (x_min * img_size, y_min * img_size, x_max * img_size, y_max * img_size)
        )
        # %.2f
        # title = f"x_min, x_max, y_min, y_max = {x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f}"
        # plt.imshow(cropped_image)
        # plt.title(title)
        # plt.show()
        cropped_image = cropped_image.resize((img_size, img_size))
        return cropped_image, keypoint

    def load_img(self, image_path, keypoint, log=False):
        
        key = image_path
        image_pil = self.load_cache(key, lambda: Image.open(image_path))

        # 0-255
        if log:
            print()
            print("-----log load img()")
            print("bef image size|mode =", image_pil.size, image_pil.mode)
            # find min max of image_pil
            # tensor from pil image
            tensor = transforms.ToTensor()(image_pil)
            mn, mx = tensor.min(), tensor.max()
            print("bef -> mn, mx =", mn, mx)

        if self.no_aug or self.dataset == "te" or self.dataset == 'va':
            key = image_path + 't'
            image_tensor = self.load_cache(key, lambda: self.do_totensor(image_pil))
        else:
            image_tensor, keypoint = self.do_transform(image_pil, keypoint)

        if log:
            # print min max of image
            mn, mx = image_tensor.min(), image_tensor.max()
            print("aft -> mn, mx =", mn, mx)

        return image_tensor, keypoint
    
    def load_cache(self, key, func):
        if key in self.cache:
            data = self.cache[key]
        else:
            data = func()
            self.cache[key] = data
        return data

    def do_totensor(self, image_pil):
        trans = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ]
        preprocess = transforms.Compose(trans)
        image_tensor = preprocess(image_pil)
        return image_tensor

    def do_transform(self, image_pil, keypoint):
        is_aug_this_img = random.random() < 0.8

        if is_aug_this_img:
            max_angle_degree = 45
            res = self.gen_rotated_data(image_pil, keypoint, max_angle_degree)
            if type(res) is tuple:
                image_pil, keypoint = res
            
            if self.mode != Const.mode_single_point_left_shoulder:
                res = self.gen_cropped_data(image_pil, keypoint, img_size=self.img_size)
                if type(res) is tuple:
                    image_pil, keypoint = res
            
            if self.mode == Const.mode_single_point_left_shoulder:
                res = self.gen_cropped_data_for_single_point(image_pil, keypoint, img_size=self.img_size)
                if type(res) is tuple:
                    image_pil, keypoint = res
        
        trans = [
            transforms.Resize(self.img_size),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        preprocess = transforms.Compose(trans)
        image_tensor = preprocess(image_pil)
        return image_tensor, keypoint

    def read_data(self, dataset):
        path = os.path.join(root, "keypoints.json")
        with open(path, "r") as f:
            data = json.load(f)

        out = []
        for k, v in data.items():
            ind = k
            img_path = v["img_path"]
            gt = int(v["gt"])
            keypoint = v["keypoint"]
            s = v["set"]
            d = Data(ind, img_path, gt, keypoint, s)
            out.append(d)
        n = len(out)
        out = [d for d in out if d.dataset == dataset]

        print()
        print(f"{dataset} = {len(out)}/{n}")
        print("example data:")
        print(out[0])
        print()
        return out

    def read_json(self, path):
        path = os.path.join(path)
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)


def test():
    import time
    img_size = 360
    data = MyDataset("tr", img_size)
    t0 = time.time()
    for d in data:
        d = d["inp"].shape
    t1 = time.time()
    print('loaded')
    print( t1-t0, 's')

    t0 = time.time()
    for d in data:
        d = d["inp"].shape
    t1 = time.time()
    print(t1-t0, 's')

    t0 = time.time()
    for d in data:
        d = d["inp"].shape
    t1 = time.time()
    print(t1-t0, 's')
    time.sleep(10)
    
if __name__ == "__main__":
    # test()
    # set seed of random number generator
    # random.seed(0)
    # torch.manual_seed(0)

    test()
    # plot()
