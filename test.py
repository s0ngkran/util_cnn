import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from model import Model
from data01 import MyDataset, Data
from utils.setting import Setting
# import torch.nn.functional as F
# import torchvision.transforms as T

# from lossfunc_to_control_covered_F_score_idea import loss_func
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser
from config import config, Const

def do_nothing(k):
    return k

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('img_size')
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 
    parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
    parser.add_argument('-d', '--device') 
    parser.add_argument('--weight') 
    parser.add_argument('-cus', '--is_custom_mode', action="store_true") 
    parser.add_argument('--pred_keypoints', action="store_true") 
    parser.add_argument('--config') 
    args = parser.parse_args()
    assert args.device in [None, 'cpu', 'cuda']
    print(args)

    if args.is_custom_mode:
        assert args.name.startswith('m')
    img_size = int(args.img_size)

    training = config()[args.config]
    is_no_links_mode = args.name.startswith('n')
    is_no_links_custom_mode = args.name.startswith('o')
    model_kwargs = {
        "raw_config": training,
        'is_custom_mode': args.is_custom_mode,
        'is_no_links_mode': is_no_links_mode,
        'is_no_links_custom_mode': is_no_links_custom_mode,
    }
    data_kwargs = {
        "raw_config": training
    }

    ############################ config ###################
    TESTING_JSON = 'te'
    BATCH_SIZE = 5 if args.batch_size is None else args.batch_size
    TRAINING_NAME = args.name
    N_WORKERS = args.n_worker if args.n_worker is not None else 10
    SAVE_FOLDER = 'save/'
    TESTING_FOLDER = 'result/'
    DEVICE = 'cuda' if args.device is None else args.device

    print('''

    preparing TEST

    ''')
    
    WEIGHT_PATH = os.path.join(SAVE_FOLDER, f'{args.name}.best') if args.weight is None else args.weight
    print('weight_path =', WEIGHT_PATH)

    print('starting...')
    for folder_name in [TESTING_FOLDER]:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    # load data
    checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device(DEVICE))
    setting = Setting(checkpoint['setting'])

    testing_set = MyDataset(TESTING_JSON, img_size, **data_kwargs)
    testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

    # load state for amp
    links = MyDataset.get_link()
    model = Model(
        setting.model.sig_point,
        setting.model.sig_link,
        links,
        img_size=img_size,
        **model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    def test():
        global model, global_loss
        model.eval()
        pred_list = []
        gt_list = []
        key_list = []
        pred_keypoints = []
        gt_keypoints = []
        n = len(testing_set_loader)
        with torch.no_grad():
            for iteration, dat in enumerate(testing_set_loader):
                inp = dat['inp'].to(DEVICE)
                output = model(inp) 

                # try
                heats = output[0]
                assert len(heats)==3
                last = heats[2]
                output_size = last.shape[-1]

                pred_func = Data.pred_from_keypoint
                if training.get('data', None) == Const.mode_single_point_left_shoulder:
                    pred_func = lambda x: 'single point'
                    print('single_point')
                pred_batch = model.get_pred(output, pred_func)
                pred_list = pred_list + pred_batch # indexes

                gt_list.extend([gt for gt in dat['gt']]) 
                key_list.extend([k for k in dat['key']])
                if args.pred_keypoints:
                    # pred_keypoints_batch = model.get_pred(output, lambda keypoints: [k/output_size for k in keypoints])
                    pred_keypoints_batch = model.get_keypoint_batch_by_scale_up(output, device=DEVICE) # tested
                    pred_keypoints = pred_keypoints + pred_keypoints_batch 
                    gt_keypoints_ = model.gt_batch_to_list(dat['keypoint'])
                    gt_keypoints = gt_keypoints + gt_keypoints_

                print('iter',iteration+1, '/', n, f'ex: pr{pred_batch[0]}_gt{dat["gt"][0]}')
                # if iteration > 10: break
            assert len(gt_list) == len(pred_list)
            out = {
                    'pred_list': pred_list,
                    'pred_keypoints': pred_keypoints,
                    'gt_keypoints': gt_keypoints,
                    'gt_list': gt_list,
                    'key_list': key_list,
                  }
            path = os.path.join(TESTING_FOLDER,f'{args.name}.res')
            torch.save(out, path)
            print('saved keypoints ->', path)
            corr = [gt==pr for gt, pr in zip(gt_list, pred_list)]
            acc = sum(corr)/len(gt_list)
        return float(acc)

    acc = test()
    print(f'acc {args.name} =', acc, '(donot use this acc, this value not scaled up. use pred_keypoints)')
