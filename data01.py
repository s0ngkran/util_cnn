from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('../../')
from dataclasses import dataclass
from typing import List
from collections import Counter

all_gt = [i for i in range(11)]
wrong_gt = [496, 497, 796, 1290, 1289, 1318, 86, 1292, 1291]

@dataclass
class Data:
    img_path: str
    key: str
    keypoint: List
    hand_side: str
    gt: int
    user: str

    def __post_init__(self):
        self.keypoint = [(float(d[0]), float(d[1])) for d  in self.keypoint]
        assert self.gt in all_gt, f'gt-{self.gt}'


class DME_1k(Dataset):
    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def read_data(self, dataset_name):
        assert dataset_name in ['training', 'validation', 'testing']
        root = '../../dme_1k/'
        json = self.read_json(os.path.join(root, f'{dataset_name}_set.json'))

        data = []
        for i, (k,v) in enumerate(json.items()):
            if i in wrong_gt: continue
            path = v['path']
            img_path = os.path.join(root, path)
            key = path.split('/')[-1].split('.')[0]
            keypoints = v['keypoint']
            hand_side = v['hand_side']
            gt = int(v['gt'])
            if gt == 11: continue
            user = v['user']
            o = Data(img_path, key, keypoints, hand_side, gt, user)
            data.append(o)
        return data

    def load_img(self, image_path, log=False):
        image = Image.open(image_path)
        # 0-255
        if log:
            print()
            print('-----log load_img()')
            print('bef image size|mode =', image.size, image.mode)

        preprocess = transforms.Compose([
            transforms.Resize(128),
            # transforms.Resize(360),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                   std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[.5,.5,.5],
            #                     std=[.5,.5,.5]),
            # range(0-255) -> range(-1,1)
        ])
        image_tensor = preprocess(image)
        if log:
            print('aft image size|min|max =', image_tensor.size(), image_tensor.min(), image_tensor.max())
            # approx -> -2 to 2
            image_array = image_tensor.numpy()
            print('min, max =',image_array.min(), image_array.max())
            print('-----end log load_img()')
            print()
        
        return image_tensor

    def __init__(self, dataset, test_mode=False, is_load_img=True):
        print('init dme 1k dataset', dataset)
        data = self.read_data(dataset)
        
        self.img_path = []
        self.img = []
        self.ground_truth = []
        self.raw = []
        
        if test_mode:
            data = data[:100]
            
        print('start load image...')
        for i, dat in enumerate(data):
            if i%100==0:
                print('...loaded', i)
            img_path = dat.img_path
            gt = dat.gt

            # load img
            if is_load_img:
                img = self.load_img(img_path, log= i <= 0)
                self.img.append(img)

            # append 
            self.ground_truth.append(gt)
            self.img_path.append(img_path)
            self.raw.append(dat)
        print(f'loaded {len(self.img)=} {len(self.img_path)=} {len(self.ground_truth)=} {dataset=}')

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'img': self.img[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans

def test():
    dataset = DME_1k('training', test_mode=True)
    dataset = DME_1k('validation', test_mode=True)
    dataset = DME_1k('testing', test_mode=True)

    # dataset = DME_1k('validation', test_mode=False, is_load_img=False)
    # print('here', len(dataset))
    # x = []
    # for d in dataset.raw:
    #     x.append(d.user)
    # print('passed')

    # x = Counter(x)
    # print(x)

if __name__ == '__main__':
    test()

