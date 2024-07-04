from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('../../')
from s1_data_model import S1DataModel

class Dataset_S1_1000(Dataset):
    def read_data(self):
        path = '/host/s1_data/s1_data/img1000/img1000.json'

        path = '/home/tatpong/skd'
        # path = path + '/s1_data/s1_data/img1000/img1000.json'
        path = path + '/s1_data/s1_data/img3000/ready_new.json'
        data = S1DataModel(json_path=path, out_jpg_dir='../../')
        return data

    def load_img(self, image_path, log=False):
        image = Image.open(image_path)
        # 0-255
        if log:
            print('bef image size|mode =', image.size, image.mode)

        preprocess = transforms.Compose([
            transforms.Resize(128),
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
        
        return image_tensor

    def __init__(self, dataset, test_mode=False):
        dataset_key = {
            'training': 'tr',
            'validation': 'va',
            'testing': 'te',
        }
        dataset = dataset_key[dataset]

        s = self.read_data()
        # assert len(s.data) == 1000
        
        allowed_gt = [str(i) for i in range(0, 11)]
        
        self.img_path = []
        self.img = []
        self.ground_truth = []
        
        tr_set = s.get_only_set('tr')
        va_set = s.get_only_set('va')
        te_set = s.get_only_set('te')
        # assert len(tr_set) == 698
        #assert len(va_set) == 153
        #assert len(te_set) == 149
        assert len(tr_set) > 698
        assert len(va_set) > 153
        assert len(te_set) > 149
        
        data = s.get_only_set(dataset)
        data = list(data.values())
        
        if test_mode:
            data = data[:100]
            
        for i, dat in enumerate(data):
            assert dat['set'] == dataset
            assert dat['gt'] in allowed_gt

            # get img path
            img_path = dat['img_path']
            # img_path = os.path.join('../../',img_path)
            img_path = os.path.join('../',img_path)
            # append img_path
            self.img_path.append(img_path)
            # load img
            img = self.load_img(img_path, log= i <= 0)

            # append img
            self.img.append(img)
            # append gt
            gt = int(dat['gt'])
            self.ground_truth.append(gt)
        print('loaded',len(self.img),len(self.img_path),len(self.ground_truth), dataset)

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'img': self.img[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans
