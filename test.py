import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from model import Model
from data01 import MyDataset
# import torch.nn.functional as F
# import torchvision.transforms as T

# from lossfunc_to_control_covered_F_score_idea import loss_func
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-nd', '--no_drop', action='store_true')
    parser.add_argument('--out11', action='store_true')
    parser.add_argument('--no_bn_dr', action='store_true')
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 
    parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
    parser.add_argument('-d', '--device') 
    args = parser.parse_args()
    assert args.device in [None, 'cpu', 'cuda']
    print(args)
    model_kwargs = {
        'no_drop': args.no_drop,
        'out11': args.out11,
        'no_bn_dr': args.no_bn_dr,
    }
    data_kwargs = {
        'no_aug': args.no_aug,
    }

    ############################ config ###################
    TESTING_JSON = 'tr'
    BATCH_SIZE = 8 if args.batch_size == None else args.batch_size
    TRAINING_NAME = args.name
    N_WORKERS = args.n_worker if args.n_worker != None else 10
    SAVE_FOLDER = 'save/'
    TESTING_FOLDER = 'result/'
    DEVICE = 'cuda' if args.device is None else args.device

    print('''

    preparing TEST

    ''')
    
    WEIGHT_PATH = os.path.join(SAVE_FOLDER, f'{args.name}best_epoch.model')
    print('weight_path =', WEIGHT_PATH)

    print('starting...')
    for folder_name in [TESTING_FOLDER]:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    # load data
    testing_set = MyDataset(TESTING_JSON, **data_kwargs)
    testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

    model = Model(**model_kwargs).to(DEVICE)
    epoch = 0
    lowest_va_loss = 9999999999
    
    # load state for amp
    checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])

    def test():
        global model, global_loss
        model.eval()
        pred_list = []
        gt_list = []
        key_list = []
        n = len(testing_set_loader)
        with torch.no_grad():
            for iteration, dat in enumerate(testing_set_loader):
                inp = dat['inp'].to(DEVICE)
                output = model(inp) 
                pred = [output[i].cpu().numpy() for i in range(len(output))]
                pred_list = pred_list + pred
                
                gt = dat['ground_truth']
                gt = [int(gt[i]) for i in range(len(gt))]
                gt_list = gt_list + gt

                key = [k for k in dat['key']]
                key_list = key_list + key
                print('iter',iteration+1, '/', n)
                # if iteration > 10: break
            assert len(gt_list) == len(pred_list)
            out = {
                    'pred_list': pred_list,
                    'gt_list': gt_list,
                    'key_list': key_list,
                  }
            torch_save(os.path.join(TESTING_FOLDER,f'{args.name}.res'), out)
            acc = get_acc(gt_list, pred_list)
        return acc

    def get_acc(gt_list, pred_list):
        correct = 0
        fail = 0
        for gt, pr in zip(gt_list, pred_list):
            pr = np.argmax(pr)
            if gt == pr:
                correct += 1
            else: 
                fail += 1
        assert correct + fail == len(gt_list)
        return correct/len(gt_list)

    def torch_save(filename, out):
        torch.save(out, filename)
        print('saved', filename)

    acc = test()
    print(f'acc {args.name} =', acc)
