import torch
import os
import sys
sys.path.append('../')
from sk_log import SKLogger
import time
from torch.utils.data import DataLoader
from model import Model
from data01 import MyDataset
from stopper import Stopper
# import torch.nn.functional as F

import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('name') 
parser.add_argument('-ck', '--checking', help='run only first 50 sample', action='store_true') 
parser.add_argument('-co', '--continue_save', help='continue at specific epoch', type=int) 
parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 

parser.add_argument('-col', '--continue_last', help='continue at last epoch', action='store_true')
parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
parser.add_argument('-d', '--device')
parser.add_argument('-nlr', '--new_learning_rate',  type=int)
parser.add_argument('-lr', '--learning_rate',  type=int)
args = parser.parse_args()
print(args)
assert args.device in [None, 'cpu', 'cuda']

############################ config ###################
TRAINING_JSON = 'tr'
VALIDATION_JSON = 'va'
BATCH_SIZE = 16 if args.batch_size is None else args.batch_size
SAVE_EVERY = 1
LEARNING_RATE = 1e-4 if args.learning_rate is None else 10**args.learning_rate
TRAINING_NAME = args.name
N_WORKERS = args.n_worker if args.n_worker is not None else 10
LOG_FOLDER = 'log/'
SAVE_FOLDER = 'save/'
CHECKING = args.checking
AMP_ENABLED = False
OPT_LEVEL = 'O2'
IS_CONTINUE = args.continue_last 
DEVICE = 'cuda' if args.device is None else args.device
NEW_LEARNING_RATE = None if args.new_learning_rate is None else 10**args.new_learning_rate
print('training name:', TRAINING_NAME)

def feed(dat):
    inp = dat['inp'].to(DEVICE)
    gt = dat['ground_truth']
    optimizer.zero_grad()
    output = model(inp)
    gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()
    loss = loss_func(output, gt)
    return loss
############################ config ###################

print('starting...')
for folder_name in [LOG_FOLDER, SAVE_FOLDER]:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# load data
training_set = MyDataset(TRAINING_JSON, test_mode=CHECKING)
validation_set = MyDataset(VALIDATION_JSON, test_mode=CHECKING)
training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)
print('tr set', len(training_set))
print('batch size', BATCH_SIZE)
assert len(training_set) >= BATCH_SIZE, 'please reduce batch size'

model = Model().to(DEVICE)
stopper = Stopper()
optimizer = torch.optim.Adam(model.parameters())
epoch = 0
lowest_va_loss = 9999999999

# load state for amp
if IS_CONTINUE:
    CONTINUE_PATH = SAVE_FOLDER + TRAINING_NAME + 'last_epoch.model'
    print('continue on last epoch')
    print(CONTINUE_PATH)
    print()
    checkpoint = torch.load(CONTINUE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # amp.load_state_dict(checkpoint['amp_state_dict'])
    epoch = checkpoint['epoch']
    lowest_va_loss = checkpoint['lowest_va_loss']
    stopper = Stopper(epoch=epoch, best_loss=lowest_va_loss)
    print('loaded epoch ->', epoch)
    if NEW_LEARNING_RATE is not None:
        print('change learning rate to 10**', NEW_LEARNING_RATE)
        # scale learning rate
        update_per_epoch = len(training_set_loader)/BATCH_SIZE
        learning_rate = NEW_LEARNING_RATE/update_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
else:
    print('\n\nlearning rate =', LEARNING_RATE)
    learning_rate = LEARNING_RATE

log = SKLogger(TRAINING_NAME, root='/host')

def train():
    global model, optimizer, epoch
    model.train()
    epoch += 1
    losses = []
    for iteration, dat in enumerate(training_set_loader):
        loss = feed(dat)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    if CHECKING:
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
        print('ep', epoch, '---loss- %.6f'%loss.item())
    return losses

def validation():
    global model, global_loss
    model.eval()
    with torch.no_grad():
        losses = []
        for iteration, dat in enumerate(validation_set_loader):
            # inp = dat['img']
            loss = feed(dat)
            if CHECKING:
                print('va loss', loss.item())
            losses.append(loss.item())

        avg_losses = sum(losses)/len(losses)
        if CHECKING:
            print('va loss', avg_losses)
            time.sleep(0.3)
            print('ep', epoch, '-----------va- %.6f'%losses)
        return losses

def torch_save(filename, out):
    torch.save(out, filename)
    print('saved', filename)
    
def avg(losses: list):
    return sum(losses) / len(losses)

def main():
    global lowest_va_loss
    # train
    while True:
        # print('fail')
        # break
        tr_losses = train()
        
        if epoch == 1 or epoch % SAVE_EVERY == 0:
            d = {
                'lowest_va_loss': lowest_va_loss,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(d, SAVE_FOLDER + TRAINING_NAME + 'last_epoch.model')

            # validate if model is saved
            va_losses = validation()
            va_loss = avg(va_losses)

            if va_loss < lowest_va_loss:
                # save best weight
                lowest_va_loss = va_loss
                torch.save(d, SAVE_FOLDER + TRAINING_NAME + 'best_epoch.model')
                print('*** saved best ep=', epoch)
            if CHECKING:
                print('* saved last ep', epoch)
            write_loss(epoch, avg(tr_losses), va_loss)
            if stopper(va_loss):
                print('breaked by stopper at ep', epoch)
                break

def write_loss(epoch, tr, va):
    log.write(epoch, tr, va)

if __name__ == '__main__':
    main()
    # python train.py name -ck