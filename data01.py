from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms 

from dataclasses import dataclass

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
    def __init__(self, dataset, test_mode=False, **kwargs):
        assert dataset in ['tr', 'va', 'te']
        print('Data kwargs:',kwargs)
        self.dataset = dataset
        data = self.read_data(dataset)
        self.no_aug = True if kwargs.get('no_aug') == True else False
        if test_mode:
            data = data[:100]
            print(f'checking mode {len(data)=}')
        self.data = data
        self.link = self.get_link()

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.load_img(img_path)
        gts = 
        gt = gts, gtl
        ans = {
            'inp': img,
            'gt': gt,
            'poh_gt': self.data[idx].gt,
            'raw': self.data[idx],
        }
        return ans
    def get_link(self):
        allowed_link = [
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
        return allowed_link

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
            dataset = v['set']
            d = Data(ind, img_path, gt, keypoint, dataset)
            out.append(d)
        n = len(out)
        out = [d for d in out if d.dataset == dataset]

        print('example data:')
        print(out[0])
        print()
        print(f'{dataset} = {len(out)}/{n}')
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
        return len(self.gt)

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


# write class to read the file "ready_new.json" to get data
class S1DataModel:
    def __init__(self, json_path='./ready_new.json', out_jpg_dir='../', img_abs_path=True, sigma_point=11.6,
    sigma_link=11.6):
        self.sigma_point = sigma_point
        self.sigma_link = sigma_link
        self.json_path = json_path
        self.data = {}
        self.data = self._read_file() 
        self.keypoints = {}
        for key in self.data.keys():
            self.keypoints[key] = self._get_keypoint(key)
        self.gaussian_map = self.gen_big_map()
        self.callable_indexes = []
        if img_abs_path:
            self.data = self._img_abs_path(self.data, out_jpg_dir)

    def _read_file(self):
        with open(self.json_path) as json_file:
            data = json.load(json_file)
        return data

    def _img_abs_path(self, data, root):
        # change path to absolute path
        # "./drive/MyDrive/MasterEN/s1_dataset/to_coke/out_jpg/0.jpg" -> "./out_jpg/0.jpg" 
        for key in data.keys():
            data[key]['img_path'] = data[key]['img_path'].replace('./drive/MyDrive/MasterEN/s1_dataset/to_coke/', root)
        return data

    def count(self):
        # count how many tr, va, te
        tr = 0
        va = 0
        te = 0
        for key in self.data.keys():
            if self.data[key]['set'] == 'tr':
                tr += 1
            elif self.data[key]['set'] == 'va':
                va += 1
            elif self.data[key]['set'] == 'te':
                te += 1
        return tr, va, te
    
    def get_only_set(self, set_name):
        assert set_name in ['tr', 'va', 'te'], "set_name must be 'tr', 'va', or 'te'"
        # get only tr data
        data = {}
        for key in self.data.keys():
            if self.data[key]['set'] == set_name:
                data[key] = self.data[key]
        return data
   
    def get_a_key(self, set_name):
        data = self.get_only_set(set_name)
        for key in data.keys():
            return key

    def _get_keypoint(self, key):
        # get keypoint at index i
        keypoint = self.data[str(key)]['keypoint']
        keypoint = keypoint.replace("'", "")
        keypoint = keypoint.split(',')
        w,h = 720, 720

        # every two elements is a point
        x = []
        y = []
        for i in range(len(keypoint)):
            if i % 2 == 0:
                x.append(float(keypoint[i])* w)
            else:
                y.append(float(keypoint[i])* h)
        return x, y

    def get_keypoint_of_set(self, set_name):
        # get keypoint of set
        data = self.get_only_set(set_name)
        keypoints = {}
        for key in data.keys():
            keypoint = self._get_keypoint(key)
            keypoints[key] = keypoint
        return keypoints
    
    def get_img_path(self, i):
        # get img_path at index i
        return self.data[str(i)]['img_path']
    
    def _gen_gaussian_map(self, width, height, sigma):
        x = np.linspace(-width / 2, width / 2, width)
        y = np.linspace(-height / 2, height / 2, height)
        xv, yv = np.meshgrid(x, y)
        '''
        if width = 6, height = 6

        x [-3.  -1.8 -0.6  0.6  1.8  3. ]

        y [-3.  -1.8 -0.6  0.6  1.8  3. ]

        xv [[-3.  -1.8 -0.6  0.6  1.8  3. ]
         [-3.  -1.8 -0.6  0.6  1.8  3. ]
         [-3.  -1.8 -0.6  0.6  1.8  3. ]
         [-3.  -1.8 -0.6  0.6  1.8  3. ]
         [-3.  -1.8 -0.6  0.6  1.8  3. ]
         [-3.  -1.8 -0.6  0.6  1.8  3. ]]

        yv [[-3.  -3.  -3.  -3.  -3.  -3. ]
         [-1.8 -1.8 -1.8 -1.8 -1.8 -1.8]
         [-0.6 -0.6 -0.6 -0.6 -0.6 -0.6]
         [ 0.6  0.6  0.6  0.6  0.6  0.6]
         [ 1.8  1.8  1.8  1.8  1.8  1.8]
         [ 3.   3.   3.   3.   3.   3. ]]
        '''
        
        gaussian_map = np.exp(-(xv ** 2 + yv ** 2) / (sigma ** 2))
        # convert to torch
        gaussian_map = torch.tensor(gaussian_map)
        return gaussian_map

    def gen_big_map(self):
        sigma_point = self.sigma_point
        size = 720 * 2
        # size = 720 
        width = size
        height = size

        # 0.00445 sec on 720px on macboo
        self.gaussian_map = self._gen_gaussian_map(width, height, sigma_point) 
        return self.gaussian_map
    
    def _get_gt_point(self, x_list, y_list, big_gaussian_map):
        tensor_gaussian_map = torch.zeros((len(x_list), 720, 720))
        for i in range(len(x_list)):
            # plot keypoint
            # plt.scatter(x_list[i], y_list[i], c='r')
            
            size = 720
            # crop gaussian map to 720x720 while centering on keypoint
            # gaus = gaus[size-int(y[i]):size*2 - int(y[i]), size-int(x[i]):size*2 - int(x[i])]
            
            xi, yi = int(x_list[i]), int(y_list[i])
            
            gaus = big_gaussian_map[size-yi:size*2 - yi, size-xi:size*2 - xi]
            # # print each
            # print(xi, yi, 'xi, yi')
            # print(size-yi, size*2 - yi, size-xi, size*2 - xi, 'size-yi, size*2 - yi, size-xi, size*2 - xi')
            # print(gaus.shape, 'gaus shape')

            # crop gaussian map to 720x720 and assign to tensor
            tensor_gaussian_map[i] = gaus
        return tensor_gaussian_map

    def gen_gt_point(self, set_name):
        assert set_name in ['tr', 'va', 'te'], "set_name must be 'tr', 'va', or 'te'"
        # get only tr data
        keypoints = self.get_keypoint_of_set(set_name)

        self.gen_big_map()
        # map for each gt point
        gt_points = {}
        n = len(keypoints)

        # for each keypoint, get gt point
        for i, key in enumerate(keypoints.keys()):
            # cal percent with 2 floating point
            percent = round((i+1)/n*100, 2)
            print(f'\r gen gt point... {set_name} {percent} %', end='')

            # todo: allow only point on hand
            x_list, y_list = keypoints[key]
            gt_point = self._get_gt_point(x_list, y_list, self.gaussian_map)
            gt_points[key] = gt_point

        self.gt_points = gt_points
        return gt_points
    
    def gen_gt_link(self, set_name):
        assert set_name in ['tr', 'va', 'te'], "set_name must be 'tr', 'va', or 'te'"
        # get only tr data
        keypoints = self.get_keypoint_of_set(set_name)
        

        # map for each gt link
        gt_links = {}
        n = len(keypoints)
        for i, key in enumerate(keypoints.keys()):
            # if i > 2: break
            # display percent
            percent = round((i+1)/n*100, 2)
            print(f'\r gen gt link... {set_name} {percent} %', end='')
            
            x_list, y_list = keypoints[key]
            gt_link = self._gen_gt_link(x_list, y_list)

            gt_links[key] = gt_link
        self.gt_links = gt_links
        return gt_links
    
    def _gen_gt_link(self, x_list, y_list):
        # define link
        allowed_link = [
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
        # 18 links from pointing_hand -> arm -> body -> arm -> palm_hand

        # get gt link
        gt_link = torch.zeros((len(allowed_link) * 2, 720, 720))
        for j, link in enumerate(allowed_link):
            # get p1, p2
            p1 = np.array([x_list[link[0]], y_list[link[0]], 1])
            p2 = np.array([x_list[link[1]], y_list[link[1]], 1])
            # generate paf
            paf = self.generate_paf(p1, p2)
            gt_link[j * 2] = paf[0]
            gt_link[j * 2 + 1] = paf[1]
        return gt_link
    
    def generate_paf(self, p1, p2):
        sigma_link = self.sigma_link
        size = 720
        paf = np.zeros((2, size, size)) # (xy, 720, 720)
        
        if p1[2] > 0 and p2[2] > 0:  # Check visibility flags
            diff = p2[:2] - p1[:2]
            # convert to unit vector
            norm = np.linalg.norm(diff)
            # print()
            # print(norm, 'norm')
            # print(diff, 'diff')
            
            # if norm > 1e-6, then diff is not zero vector
            if norm > 1e-6:
                # unit vector
                v = diff / norm
                v_perpendicular = np.array([-v[1], v[0]])

                # meshgrid
                x, y = np.meshgrid(np.arange(size), np.arange(size))

                dist_x = x - p1[0]
                dist_y = y - p1[1]

                dist_along = v[0] *  dist_x + v[1] * dist_y
                dist_perpendicular = np.abs(v_perpendicular[0] * dist_x + v_perpendicular[1] * dist_y)
                
                # mask distance
                mask1 = dist_along >= 0
                mask2 = dist_along <= norm
                mask3 = dist_perpendicular <= sigma_link
                mask = mask1 & mask2 & mask3

                # add unit vector to paf_x and paf_y
                paf[0, mask] = v[0]
                paf[1, mask] = v[1]
        # convert to torch
        paf = torch.tensor(paf)
        return paf 

    def gen_gt(self, key):
        x_list, y_list = self.keypoints[key]
        # gen both gt point and gt link
        gt_point = self._get_gt_point(x_list, y_list, self.gaussian_map)
        gt_link = self._gen_gt_link(x_list, y_list)
        
        # print()
        # print(gt_point.shape, 'gt_point shape')
        # print()
        # print(gt_link.shape, 'gt_link shape')
        # print()
        return (gt_point, gt_link)

    def gen_gt_link_from_key(self, key):
        x_list, y_list = self.keypoints[key]
        gt_link = self._gen_gt_link(x_list, y_list)
        return gt_link
    def gen_gt_point_from_key(self, key):
        x_list, y_list = self.keypoints[key]
        gt_point = self._get_gt_point(x_list, y_list, self.gaussian_map)
        return gt_point

    def plot(self, i):
        if 'plt' not in globals():
            print()
            print("!!!WARNING no matplotlib.pyplot")
            return

        # plot img using pyplot at index i
        img_path = self.get_img_path(i)
        img = plt.imread(img_path)
        plt.imshow(img)

        x, y = self._get_keypoint(i)

        # plot
        plt.scatter(x, y, c='r')
        plt.show()
