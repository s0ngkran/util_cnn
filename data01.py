from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms 

from dataclasses import dataclass

@dataclass 
class Data:
    gt: str
    img_path: str
    key: str = ''
    
    def __post_init__(self):
        root = '../poh_plain_vs_natural'
        self.key = self.img_path.split('/')[-1]
        self.img_path = os.path.join(root, self.img_path)
        out = self.my_gt()
        self.gt = out[self.gt]
    def my_gt(self):
        all_gt = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        out = {}
        for i, gt in enumerate(all_gt):
            out[gt] = i
        return out


class MyDataset(Dataset):
    def __init__(self, dataset, test_mode=False, **kwargs):
        assert dataset in ['tr', 'va', 'te']
        print('Data kwargs:',kwargs)
        dataset = {
            'tr': 'training',
            'va': 'validation',
            'te': 'testing'
        }[dataset]
        self.dataset = dataset
        data = self.read_data(dataset)
        self.no_aug = True if kwargs.get('no_aug') == True else False

        if test_mode:
            data = data[:100]
            print(f'checking mode {len(data)=}')
            
        self.ground_truth = []
        self.img_path = []
        self.key = []
        for i, dat in enumerate(data):
            self.ground_truth.append(dat.gt)
            self.img_path.append(dat.img_path)
            self.key.append(dat.key)
        print(f'loaded {len(self.ground_truth)=}')

    def __getitem__(self, idx):
        path = self.img_path[idx]
        img = self.load_img(path)
        gt  =self.ground_truth[idx]
        key = self.key[idx]
        ans = {
            'inp': img,
            'ground_truth': gt,
            'key': key,
        }
        return ans

    def load_img(self, image_path, log=False):
        image = Image.open(image_path)
        # 0-255
        if log:
            print()
            print('-----log load img()')
            print('bef image size|mode =', image.size, image.mode)

        img_size = 220
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
        root = '../poh_plain_vs_natural/processed'
        path = os.path.join(root, f'poh_black_{dataset}.json')
        with open(path, 'r') as f:
            data = json.load(f)
        out = []
        for d in data:
            d = Data(d['gt'], d['img_path'])
            out.append(d)
        print('example data:')
        print(d)
        print()
        print(f'{dataset}_set {len(out)=}')
        return out

    def read_json(self, path):
        path = os.path.join(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.ground_truth)

def test():
    data = MyDataset('tr', **{'no_aug': True})
    dd = {}
    for d in data:
        print(d['inp'].shape, d['ground_truth'])
        dd[d['ground_truth']] = 1
        break
    pass

if __name__ == "__main__":
    test()
