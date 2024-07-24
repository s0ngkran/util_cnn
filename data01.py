from torch.utils.data import Dataset
import os
import json

from dataclasses import dataclass

@dataclass 
class Data:
    gt: str
    img_path: str

class MyData(Dataset):
    def __init__(self, dataset, test_mode=False):
        assert dataset in ['tr', 'va', 'te']
        data = self.read_data()

        if test_mode:
            data = data[:100]
            print(f'test mode {len(data)=}')
            
        self.ground_truth = []
        self.keypoint = []
        for i, dat in enumerate(data):
            self.ground_truth.append(dat.gt)
            self.keypoint.append(dat.keypoints)
        print(f'loaded {len(self.ground_truth)=}')

    def __getitem__(self, idx):
        ans = {
            'inp': self.keypoint[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans

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
    def read_json(self, path):
        path = os.path.join(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.ground_truth)

def test():
    # data = TFSSingleHand('tr', 'abs')
    # print()
    # for d in data:
    #     print(d)
    #     break
    pass

if __name__ == "__main__":
    test()
