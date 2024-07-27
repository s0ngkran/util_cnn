import torch
import sys
import numpy as np

def eval(result_path):
    # print('path=', result_path)
    data = torch.load(result_path)
    pred_list = data['pred_list']
    gt_list = data['gt_list']
    assert len(pred_list) == len(gt_list)

    correct = 0
    n = 0
    for gt, pred in zip(gt_list, pred_list):
        pred_ind = np.argmax(pred)
        if int(pred_ind) == int(gt):
            correct += 1
        n += 1
    print(f'acc={correct/n*100:.2f}%')

if __name__ == "__main__":
    eval(sys.argv[1])
