import torch
import time
try:
    import matplotlib.pyplot as plt
except:
    pass

class GTGen:
    def __init__(self, img_size, sigma_points, sigma_links, links, n_keypoint=19):
        self.img_size = img_size
        self.links = links
        self.n_keypoint = n_keypoint
        self.sigma_points = sigma_points
        self.sigma_links = sigma_links
        gaussian_size = img_size *2
        big_gaussians = {}
        for sigma_point in sigma_points:
            if str(sigma_point) in big_gaussians:
                continue
            m = self._gen_gaussian_map(gaussian_size, gaussian_size, sigma_point)
            big_gaussians[str(sigma_point)] = m
        self.big_gaussians = big_gaussians
    
    def __call__(self, keypoints):
        keypoints = self._handle_keypoint_batch(keypoints)
        batch = []
        for keypoint in keypoints:
            gt_list = self._gen_one_img(keypoint)
            batch.append(gt_list)
        return batch
    def time(self, keypoints):
        t1 = time.time()
        keypoints = self._handle_keypoint_batch(keypoints)
        t2 = time.time()
        batch = []
        for keypoint in keypoints:
            assert len(keypoint) == self.n_keypoint, f'{len(keypoint)}'
            gt_list = self._gen_one_img(keypoint)
            batch.append(gt_list)
        t3 = time.time()
        print(t2-t1, 'handle keypoint')
        print(t3-t2, 'gen gt')
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


    def _gen_one_img(self, keypoint):
        # t0 = time.time()
        size = self.img_size
        x_list = [k[0]*size for k in keypoint]
        y_list = [k[1]*size for k in keypoint]
        xy = x_list, y_list
        gt_list = []
        for sp, sl in zip(self.sigma_points, self.sigma_links):
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
        gtl = self._gen_gtl(x_list, y_list, sigma_link)
        return (gts, gtl)

    def _gen_gaussian_map(self, width, height, sigma):
        x = torch.linspace(-width / 2, width / 2, width)
        y = torch.linspace(-height / 2, height / 2, height)
        # already add indexing but the WARNING still remains
        try:
            xv, yv = torch.meshgrid(x, y, indexing='ij')
        except:
            xv, yv = torch.meshgrid(x, y)

        gaussian_map = torch.exp(-(xv ** 2 + yv ** 2) / (sigma ** 2))
        # print(gaussian_map.shape) # == (width, hight)
        return gaussian_map

    def _gen_gts(self, x_list, y_list, sigma_point, **kw):
        big_gaussian_map = self.big_gaussians[str(sigma_point)]
        size = self.img_size
        tensor_gaussian_map = torch.zeros((len(x_list), size, size))
        for i in range(len(x_list)):
            # crop gaussian map by centering on keypoint
            xi, yi = int(x_list[i]), int(y_list[i])
            # print(xi, yi)
            gaus = big_gaussian_map[size-yi:size*2 - yi, size-xi:size*2 - xi]
            tensor_gaussian_map[i] = gaus
        return tensor_gaussian_map

    def _gen_gtl(self, x_list, y_list, sigma_link, **kw):
        links = self.links
        size = self.img_size
        gt_link = torch.zeros((len(links) * 2, size, size))
        for i, (p1, p2) in enumerate(links):
            # generate paf
            p1 = torch.tensor([y_list[p1], x_list[p1], 1])
            p2 = torch.tensor([y_list[p2], x_list[p2], 1])
            paf = self._generate_paf(p1, p2, size, sigma_link)
            gt_link[i * 2] = paf[0]
            gt_link[i * 2 + 1] = paf[1]
        return gt_link

    def _generate_paf(self, p1, p2, size, sigma_link):
        paf = torch.zeros((2, size, size)) # (xy, 720, 720)
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

                dist_along = v[0] *  dist_x + v[1] * dist_y
                dist_perpendicular = torch.abs(v_perpendicular[0] * dist_x + v_perpendicular[1] * dist_y)
                
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
        print('test gen gts')

        sigma = 11.6
        keypoint = [(0.1,0.3),(0.4,0.1)]
        size = 720
        x_list = [k[0]*size for k in keypoint]
        y_list = [k[1]*size for k in keypoint]
        gts = self._gen_gts(x_list, y_list, sigma)
        assert gts.shape == torch.Size([len(keypoint),size,size]), f'{gts.shape}'
        plt.imshow(gts[0])
        plt.show()

    def test_gen_gtl(self):
        print('test gen gtl')
        size, sigma = 720, 11.6

        keypoint = [(0.1,0.3),(0.4,0.1)]
        x_list = [k[0]*size for k in keypoint]
        y_list = [k[1]*size for k in keypoint]
        gtl = self._gen_gtl(x_list, y_list, sigma)
        assert gtl.shape == torch.Size([len(keypoint),size,size]), f'{gtl.shape}'
        plt.imshow(gtl[0])
        plt.show()

def test_mini():
    size=720
    sigma_points=[11.6]
    links=[(0,1)]
    g = GTGen(size, sigma_points, links)
    g.test_gen_gtl()
    g.test_gen_gts()
    print('passed gts gtl')

def test_gen_gt():
    size = 64
    keypoint = [[[.1],[.3]],[[.4],[.1]],[[.5],[.6]]]
    # 19 2 5
    link = [(0,1)]
    sigma_points = [11.6, 7.6, 4.6]
    sigma_links = [11.6, 7.6, 4.6]

    gen = GTGen(size,  sigma_points, sigma_links, link)
    gt_list = gen(keypoint)

    n = len(gt_list)
    fig, axs =  plt.subplots(n * 2)
    for i, gt in enumerate(gt_list):
        gts, gtl = gt
        print(gts.shape)
        print(gtl.shape)
        print('--')
        axs[i].imshow(gts[0])
        axs[i+n].imshow(gtl[0])
    plt.show()


if __name__ == "__main__":
    # test_mini()
    test_gen_gt()
