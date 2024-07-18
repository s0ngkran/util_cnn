from torch.utils.data import Dataset
import os
import json

import sys
sys.path.append('../../')

from dataclasses import dataclass

@dataclass
class SingleHand:
    root: str
    img_path: str
    gt: str
    set_name: str            
    key: str = ''
    keypoints: str = ''
    
    def __post_init__(self):
        self.key = self.img_path.split('.')[0]
        self.img_path = os.path.join(self.root, 'images', self.img_path)
        
    def get_relative_vector(self, wrist, p):
        vec = p[0] - wrist[0], p[1] - wrist[1]
        return vec
    
    def set_keypoint(self, kp, mode):
        assert mode in ['abs', 'rel']
        w, h = 1280, 720
        keypoints = [(float(x)/w, float(y)/h) for x,y,_ in kp]
        
        if mode == 'abs':
            self.keypoints = keypoints
            return
        
        wrist = keypoints.pop(0)
        assert len(keypoints) == 21 -1
        keypoints = [self.get_relative_vector(wrist, p) for p in keypoints]
        self.keypoints = keypoints

class TFSSingleHand(Dataset):
    def __init__(self, dataset, absolute_or_relative, test_mode=False):
        assert dataset in ['tr', 'va', 'te']
        assert absolute_or_relative in ['abs', 'rel']

        data = self.read_data(mode=absolute_or_relative)
        n = len(data)
        data = [d for d in data if d.set_name == dataset]
        print(f'{dataset} is loaded; {len(data)=}; {len(data)/n*100:.2f}%')

        if test_mode:
            data = data[:100]
            print(f'test mode {len(data)=}')
        gt_map = self.get_gt_map()
            
        self.ground_truth = []
        self.keypoint = []
        for i, dat in enumerate(data):
            gt = gt_map[dat.gt]
            self.ground_truth.append(gt)
            self.keypoint.append(dat.keypoints)
        print(f'loaded {len(self.ground_truth)=}')

    def __getitem__(self, idx):
        ans = {
            'inp': self.keypoint[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans
    def get_gt_map(self):
        data = {
            "1": "A",
            "2": "B",
            "3": "D",
            "4": "F",
            "5": "H",
            "6": "K",
            "7": "L",
            "8": "M",
            "9": "N",
            "10": "P",
            "11": "R",
            "12": "S",
            "13": "T",
            "14": "W",
            "15": "Y",
            "16": "1",
            "17": "2",
            "18": "3",
            "19": "4",
            "20": "5",
            "21": "G",
            "22": "C",
            "23": "S+",
            "24": "J",
            "25": "J+",
            "26": "E",
            "27": "O",
            "28": "O+",
            "29": "I",
            "30": "I+"
        }
        gt_list = []
        for k,v in data.items():
            gt_list.append(v)
        gt_map = {}
        for i, gt in enumerate(gt_list):
            gt_map[gt] = i
        return gt_map

    def read_data(self, mode):
        root = '../data_zip/TFS_single_hand'
        path = os.path.join(root, 'gt.json')
        with open(path, 'r') as f:
            data = json.load(f)
        keypoints = self.read_keypoints()
        
        out = []
        no_kp = []
        for k, dat in data.items():
            s = SingleHand(root, dat['img_path'], dat['gt'], dat['set_name'])
            if s.key not in keypoints: continue
            kp = keypoints[s.key]
            if len(kp) != 21:
                no_kp.append(s.key)
                continue
            s.set_keypoint(kp, mode)
            out.append(s)
        
        # verify all outs have keypoints
        for o in out:
            fixed = 20 if mode == 'rel' else 21
            assert len(o.keypoints) == fixed, f'{len(o.keypoints)}'
        
        print(f'{len(out)=} {len(no_kp)=}')
        return out
    def read_keypoints(self):
        path = os.path.join('mph_keypoints.json')
        with open(path, 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.ground_truth)


def test():
    data = TFSSingleHand('tr', 'abs')
    print()
    for d in data:
        print(d)
        break
    data = TFSSingleHand('tr', 'rel')
    print()
    for d in data:
        print(d)
        break
    print()

if __name__ == "__main__":
    test()
