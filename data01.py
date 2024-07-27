from torch.utils.data import Dataset
import os
import time
import json
from PIL import Image
from torchvision import transforms 
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass 
class Data:
    ind: str
    img_path: str
    gt: int
    keypoint: list
    dataset: str 
    key: str = ''
    
    def __post_init__(self):
        root = '../poh_vr_1k'    
        self.key = self.img_path.split('/')[-1]
        self.img_path = os.path.join(root, self.img_path)
        assert type(self.gt) == int
        assert self.gt >= 0 and self.gt <= 10
        assert len(self.keypoint) == 19
        assert self.dataset in ['tr', 'va', 'te']


class MyDataset(Dataset):
    def __init__(self, dataset, img_size, test_mode=False, **kwargs):
        assert dataset in ['tr', 'va', 'te']
        self.dataset = dataset
        self.img_size = img_size
        data = self.read_data(dataset)
        self.no_aug = True if kwargs.get('no_aug') == True else False
        if test_mode:
            data = data[:100]
            print(f'checking mode {len(data)=}')
        self.data = data
        self.link = self.get_link()

    def __getitem__(self, idx):
        data = self.data[idx]
        img_path = data.img_path
        img = self.load_img(img_path)
        ans = {
            'inp': img,
            'keypoint': data.keypoint,
            # 'poh_gt': data.gt,
        }
        return ans

    @staticmethod
    def get_link():
        links = [
            [0,1],
            [1,2],
            [2,3],
            [3,4],
            [4,5],
            [5,6],
            [6,7],   
            [7,8],
            [8,9],
            [7,18],
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

    def load_img(self, image_path,  log=False):
        image = Image.open(image_path)
        # 0-255
        if log:
            print()
            print('-----log load img()')
            print('bef image size|mode =', image.size, image.mode)

        img_size = self.img_size
        preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomRotation(5),        
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1), ratio=(1.0, 1.0)), 
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
        ])
        if self.no_aug or self.dataset == 'testing':
            preprocess = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        image_tensor = preprocess(image)
        if log:
            print('aft image size|min|max =', image_tensor.size(), image_tensor.min(), image_tensor.max())
            # approx -> -2 to 2
            image_array = image_tensor.numpy()
            print('min, max =',image_array.min(), image_array.max())
            print('-----end log load img()')
            print()
        
        return image_tensor

    def read_data(self, dataset):
        root = '../poh_vr_1k/'
        path = os.path.join(root, f'keypoints.json')
        with open(path, 'r') as f:
            data = json.load(f)

        out = []
        for k,v in data.items():
            ind = k
            img_path = v['img_path']
            gt = int(v['gt'])
            keypoint = v['keypoint']
            s = v['set']
            d = Data(ind, img_path, gt, keypoint, s)
            out.append(d)
        n = len(out)
        out = [d for d in out if d.dataset == dataset]

        print()
        print(f'{dataset} = {len(out)}/{n}')
        print('example data:')
        print(out[0])
        print()
        return out

    def read_json(self, path):
        path = os.path.join(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    

    def gen_big_map(self):
        sigma_point = self.sigma_point
        size = 720 * 2
        # size = 720 
        width = size
        height = size

        # 0.00445 sec on 720px on macboo
        self.gaussian_map = self._gen_gaussian_map(width, height, sigma_point) 
        return self.gaussian_map

    def __len__(self):
        return len(self.data)

def test():
    img_size = 360
    data = MyDataset('va', img_size)
    for d in data:
        d = d['inp'].shape
        print(d)
        break

    times = []
    for i in range(1,10,2):
        print(i)
        t1 = time.time()
        loader = DataLoader(data, batch_size=5, num_workers=i)
        for d in loader:
            d  =d['inp'].shape
            # print(d)
            continue
        t2 = time.time()
        t = t2-t1
        times.append((i, t))

    for i, t in times:
        print(i, t)

if __name__ == "__main__":
    test()

