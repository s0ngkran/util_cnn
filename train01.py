
import torch
import os
import numpy as np
import json
import time
import sys
from torch.utils.data import DataLoader
from model01 import VGG16 as Model
from data01 import DME_1k as MyDataset
import torch.nn.functional as F

# from lossfunc_to_control_covered_F_score_idea import loss_func
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser

if __name__ == '__main__':
    '''
    python train01.py -sh -ck
    nohup python train01.py > 1tr.log&
    '''

    parser = ArgumentParser()
    parser.add_argument('-sh', '--show', help='show loss value on training and validation', action='store_true')
    parser.add_argument('-ck', '--check_run', help='run only first 50 sample', action='store_true') 
    parser.add_argument('-co', '--continue_save', help='continue at specific epoch', type=int) 
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 
    parser.add_argument('-te', '--test', help='test for s1 vgg', action="store_true") 

    parser.add_argument('-bt', '--bootstrap', help='train with bootstrap', action='store_true')
    parser.add_argument('-col', '--continue_last', help='continue at last epoch', action='store_true')
    parser.add_argument('-xx', '--xx', help='train with bootstrap', action='store_true')
    parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
    parser.add_argument('-lr', '--lr', help='lr', type=int)
    args = parser.parse_args()
    print(args)

    ############################ config ###################
    JSON_PATTERN = 'XXX'
    TRAINING_JSON = JSON_PATTERN.replace('XXX', 'training')
    VALIDATION_JSON = JSON_PATTERN.replace('XXX', 'validation')
    BATCH_SIZE = 64 if args.batch_size == None else args.batch_size
    SAVE_EVERY = 1
    LEARNING_RATE = 1e-4 if args.lr == None else 10**args.lr
    TRAINING_NAME = os.path.basename(__file__)
    N_WORKERS = args.n_worker if args.n_worker != None else 10
    LOG_FOLDER = 'log/'
    SAVE_FOLDER = 'save/'
    OPT_LEVEL = 'O2'
    CHECK_RUN = args.check_run
    IS_BOOTSTRAP = args.bootstrap
    AMP_ENABLED = False
    TESTING_FOLDER = 'TESTING_FOLDER/'
    IS_TEST_MODE = args.test if args.test is not None else False
    print('lr=', LEARNING_RATE)

    # continue training
    IS_CONTINUE = False if args.continue_save is None and args.continue_last == False else True
    # CONTINUE_PATH = './save/train09.pyepoch0000003702.model'
    if args.continue_last == False:
        continue_epoch = args.continue_save
        CONTINUE_PATH = './%s/%sepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME,  str(continue_epoch).zfill(10))
    else:
        def find_last_epoch_path():
            for _,_,fname_list in os.walk(SAVE_FOLDER):
                print('walk')
            
            name_list = []
            for fname in fname_list:
                if fname.startswith(TRAINING_NAME):
                    name_list.append(fname)
            name_list.sort()
            path = os.path.join(SAVE_FOLDER, name_list[-1])
            return path
        # CONTINUE_PATH = find_last_epoch_path()
        CONTINUE_PATH = SAVE_FOLDER + TRAINING_NAME + 'last_epoch.model'
        print('continue on last epoch')
        print(CONTINUE_PATH)
        print()

    IS_CHANGE_LEARNING_RATE = False
    NEW_LEARNING_RATE = 1e-3

    # check result
    if args.test:
        print('''

        preparing TEST

        ''')
        print('args.test',args.test)
        
        # check type
        try:
            pass
        except:
            pass

        # TESTING_JSON = JSON_PATTERN.replace('XXX', args.test[1]) if args.test is not None else 'nothing'
        TESTING_JSON = 'testing'
        print(TESTING_JSON,'-----as testing_set')
        DEVICE = 'cpu'
        # WEIGHT_PATH = './save/train09.pyepoch0000003702.model'
        WEIGHT_PATH = os.path.join('./save', __file__ + 'best_epoch.model')
        print('weight_path =', WEIGHT_PATH)

    #if args.test is not None:
    #    WEIGHT_PATH = './%s/%sepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME, str(args.test[0]).zfill(10))
    ############################################################

    print('starting...')
    for folder_name in [LOG_FOLDER, SAVE_FOLDER, TESTING_FOLDER]:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    if not IS_TEST_MODE and AMP_ENABLED:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            # from apex.multi_tensor_apply import multi_tensor_applier
            amp
            print('success amp')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example.")

    # manage batch
    def my_collate(batch):
        image_path, ground_truth, hand_landmarks = [],[],[]
        for item in batch:
            image_path.append(item['img_path'])
            ground_truth.append(item['ground_truth'])
            # hand_landmarks.append(item['hand_landmarks'])

        ans = {
            'img_path':image_path, 
            'ground_truth':ground_truth, 
            # 'hand_landmarks': hand_landmarks,
        }
        return ans
    
    # load data
    if not IS_TEST_MODE:
        training_set = MyDataset(TRAINING_JSON, test_mode=CHECK_RUN)
        validation_set = MyDataset(VALIDATION_JSON, test_mode=CHECK_RUN)
        training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
        validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)
        print('tr set', len(training_set))
        print('batch size', BATCH_SIZE)
        assert len(training_set) >= BATCH_SIZE, 'please reduce batch size'
    else:
        print('''

        you are running TEST

this is only for VGG testing
auto batch size -> 1
       

        ''')
        BATCH_SIZE = 1
        testing_set = MyDataset(TESTING_JSON, test_mode=CHECK_RUN)
        testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

    if not IS_TEST_MODE:
        model = Model().to('cuda')
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 0
        lowest_va_loss = 9999999999
    else:
        model = Model()
        epoch = 0
        lowest_va_loss = 9999999999
    
    # load state for amp
    if not IS_TEST_MODE:
        print('here')
        if IS_CONTINUE:
            checkpoint = torch.load(CONTINUE_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # amp.load_state_dict(checkpoint['amp_state_dict'])
            epoch = checkpoint['epoch']
            lowest_va_loss = checkpoint['lowest_va_loss']
            print('loaded epoch ->', epoch)
            if IS_CHANGE_LEARNING_RATE:
                # scale learning rate
                update_per_epoch = len(training_set_loader)/BATCH_SIZE
                learning_rate = NEW_LEARNING_RATE/update_per_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
        else:
            if False:
                # scale learning rate
                print()
                print()
                print('scale learning rate')
                print()
                print()
                print(len(training_set_loader), BATCH_SIZE, 'tr size, batch size')
                update_per_epoch = len(training_set_loader)/BATCH_SIZE
                print('upd', update_per_epoch)
                learning_rate = LEARNING_RATE/update_per_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            else:
                print('\n\nlearning rate =', LEARNING_RATE)
                learning_rate = LEARNING_RATE


        if AMP_ENABLED:
            # init amp
            print('initing... amp')
            model, optimizer = amp.initialize(model, optimizer, opt_level=OPT_LEVEL)
    else:
        checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])



    # write loss value
    def write_loss_txt(epoch, tr, va):
        with open(LOG_FOLDER + TRAINING_NAME + '.tr_va_loss', 'a') as f:
            txt = ';'.join(['t-ep-tr-va',str(int(time.time())), str(epoch), str(tr), str(va), '\n'])
            f.write(txt)
    def write_loss(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))

    def get_first_random_from_loader(loader):
        for dat in training_set_loader:
            break
        return dat

    # train
    def train():
        global model, optimizer, epoch
        model.train()
        epoch += 1
        for iteration, dat in enumerate(training_set_loader):
            if IS_BOOTSTRAP:
                dat = get_first_random_from_loader(training_set_loader)
            iteration += 1
            inp = dat['img'].cuda()
            # inp = dat['img']
            
            optimizer.zero_grad()
            output = model(inp)

            gt = dat['ground_truth']
            gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()

            loss = loss_func(output, gt)
            if CHECK_RUN:
                pass
            if AMP_ENABLED:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        if CHECK_RUN:
            print('ep', epoch, 'loss',loss.item())
            print('''
            ################################
            #                              #
            #                              #
            #    this is checking mode     #
            #                              #
            #                              #
            ################################
            ''')
        if args.show:
            print('ep', epoch, '---loss- %.6f'%loss.item())
        return loss

    def validation():
        global model, global_loss
        model.eval()
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            for iteration, dat in enumerate(validation_set_loader):
                iteration += 1
                inp = dat['img'].cuda()
                # inp = dat['img']
                
                optimizer.zero_grad()
                output = model(inp)

                gt = dat['ground_truth']
                gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()


                loss_ = loss_func(output, gt)

                if CHECK_RUN:
                    print('loss_', loss_.item())
                loss.append(loss_)

            loss = sum(loss)/len(loss)
            # write_loss_va(epoch, iteration, loss)
            if CHECK_RUN:
                print('va loss', loss.item())
                time.sleep(0.3)
            if args.show:
                print('ep', epoch, '-----------va- %.6f'%loss)
            return loss

    def calculate_ap(recall, precision):
        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i+1])
        indices = np.where(recall[:-1] != recall[1:])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        return ap

    def calculate_map(ground_truth, predictions):
        num_classes = ground_truth.shape[1]
        ap_sum = 0

        for class_idx in range(num_classes):
            gt_class = ground_truth[:, class_idx]
            pred_class = predictions[:, class_idx]
            
            sorted_indices = np.argsort(pred_class)[::-1]
            sorted_gt = gt_class[sorted_indices]
            sorted_pred = pred_class[sorted_indices]
            
            true_positive = np.cumsum(sorted_gt)
            precision = true_positive / np.arange(1, len(sorted_gt) + 1)
            recall = true_positive / np.sum(sorted_gt)
            
            ap_sum += calculate_ap(recall, precision)
        
        mAP = ap_sum / num_classes
        return mAP

    def gen_one_hot(gt_list):
        num_classes = gt_list.max() + 1
        one_hot = torch.eye(num_classes)[gt_list]
        return one_hot

    def test():
#hoo
        global model, global_loss
        model.eval()
        pred_list = []
        gt_list = []
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            data = {}
            for iteration, dat in enumerate(testing_set_loader):
                iteration += 1
                inp = dat['img']
                assert inp.shape[0] == 1
                
                output = model(inp) 
                pred_list.append(output[0].numpy())
                
                gt = dat['ground_truth']
                gt_list.append(int(gt[0]))
                print('iter',iteration)
                # if iteration > 10: break
            assert len(gt_list) == len(pred_list)
            out = {
                    'pred_list': pred_list,
                    'gt_list': gt_list,
                  }
            write_json('result.json', out)
            print('done')
            1/0

            correct = 0
            fail = 0
            for gt, pr in zip(gt_list, pred_list):
                pr = np.argmax(pr)
                if gt == pr: correct += 1
                else: fail += 1
            assert correct + fail == len(gt_list)
            print('acc=', correct/len(gt_list))
            gt_list = gen_one_hot(torch.tensor(gt_list))
            pred_list = np.array(pred_list)
            mAP = calculate_map(gt_list.numpy(), pred_list)
            print('mAP=',mAP)
            1/0 

    def write_json(filename, out):
        assert filename.endswith('.json')
        with open(filename, 'w') as f:
            json.dump(data, f)
        print('writed', filename)

    def test_write():
        global model, global_loss
        model.eval()
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            data = {}
            for iteration, dat in enumerate(testing_set_loader):
                iteration += 1
                inp = dat['img']
                assert inp.shape[0] == 1
                # inp = dat['img']
                
                # optimizer.zero_grad()
                output = model(inp) 
                pred_list.append(output[0].numpy())
                
                gt = dat['ground_truth']
                gt_list.append(int(gt[0]))
                gt_list = gen_one_hot(gt_list)
                mAP = calculate_map(gt_list, pred_list)
                1/0 
                # how to gen one hot 
                # how to push to array 

                argmax = torch.argmax(output, 1).tolist()[0]
                key = dat['img_path'][0].split('/')[-1].replace('.jpg', '')
                val = argmax
                gt = dat['ground_truth']
                gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()
                data[key + '_pred'] = str(val)
                data[key + '_gt'] = str(gt[0].item())
                


                print('iteration =',iteration)
                continue


                loss_ = loss_func(output, gt)

                if CHECK_RUN:
                    print('loss_', loss_.item())
                loss.append(loss_) 
            return
    
            print('---tested')
            print()
            print(data)
            print()
            outfile = __file__.replace('.py','') + '_output.json'
            with open(outfile, 'w') as f:
                json.dump(data, f)
            print()
            print('writed', outfile)
            print('----------------')
            return


            loss = sum(loss)/len(loss)
            # write_loss_va(epoch, iteration, loss)
            if CHECK_RUN:
                print('test loss', loss.item())
                time.sleep(0.3)
            if args.show:
                print('ep', epoch, '-----------va- %.6f'%loss.item())
            return loss

    # train
    while True:
        if not IS_TEST_MODE:
            # print('fail')
            # break
            tr_loss = train()
            
            if epoch == 1 or epoch % SAVE_EVERY == 0 and not IS_TEST_MODE:
                d = {
                    'lowest_va_loss': lowest_va_loss,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if AMP_ENABLED:
                    d['amp_state_dict']= amp.state_dict()
                torch.save(d, SAVE_FOLDER + TRAINING_NAME + 'last_epoch.model')


                # validate if model is saved
                va_loss = validation()

                if va_loss < lowest_va_loss:
                    # save best weight
                    lowest_va_loss = va_loss
                    torch.save(d, SAVE_FOLDER + TRAINING_NAME + 'best_epoch.model')
                    print('*** saved best ep=', epoch)
                if CHECK_RUN or args.show:
                    print('* saved last ep', epoch)
                write_loss_txt(epoch, tr_loss.item(), va_loss.item())
                
        else:
            print('''
running test function

''')
            test()
            break


