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
from config import config


parser = ArgumentParser()
parser.add_argument('name') 
parser.add_argument('--config')
#
parser.add_argument('-p', '--profile', action='store_true')
parser.add_argument('-ck', '--checking', help='run only first 50 samples', action='store_true') 
parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 

parser.add_argument('-col', '--continue_last', help='continue at last epoch', action='store_true')
parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
parser.add_argument('-d', '--device')
parser.add_argument('-nlr', '--new_learning_rate',  type=int)
parser.add_argument('-lr', '--learning_rate',  type=int)
parser.add_argument('-s', '--stopper_min_ep',  type=int)
args = parser.parse_args()
print(args)

training = config()[args.config]
params = {
    'name':args.config,
    'config': training,
}


sigma_points = training.get('sigma_points')
sigma_links = training.get('sigma_links')
img_size = training.get('img_size')

assert args.device in [None, 'cpu', 'cuda']
model_kwargs = {
}
data_kwargs = {
}
############################ config ###################
TRAINING_JSON = 'tr'
VALIDATION_JSON = 'va'
BATCH_SIZE = 5 if args.batch_size is None else args.batch_size
SAVE_EVERY = 1
LEARNING_RATE = 1e-3 if args.learning_rate is None else 10**args.learning_rate
TRAINING_NAME = args.name
N_WORKERS = 10 if args.n_worker is None else args.n_worker
SAVE_FOLDER = 'save/'
CHECKING = args.checking
AMP_ENABLED = False
OPT_LEVEL = 'O2'
IS_CONTINUE = args.continue_last 
DEVICE = 'cuda' if args.device is None else args.device
NEW_LEARNING_RATE = None if args.new_learning_rate is None else 10**args.new_learning_rate
MIN_STOP = 20 if args.stopper_min_ep is None else args.stopper_min_ep
print('training name:', TRAINING_NAME)

def feed(dat):
    inp = dat['inp'].to(DEVICE)
    keypoint = dat['keypoint']
    optimizer.zero_grad()
    output = model(inp)
    loss = model.cal_loss(output, keypoint, DEVICE)
    return loss
############################ config ###################

print('starting...')
for folder_name in [SAVE_FOLDER]:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

# load data
training_set = MyDataset(TRAINING_JSON, img_size, test_mode=CHECKING, **data_kwargs)
validation_set = MyDataset(VALIDATION_JSON, img_size, test_mode=CHECKING, **data_kwargs)
training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

assert len(training_set) >= BATCH_SIZE, f'batch={BATCH_SIZE}; please reduce batch size'

links = MyDataset.get_link()
model = Model(
    sigma_points,
    sigma_links,
    links,
    img_size=img_size,
    **model_kwargs).to(DEVICE)
stopper = Stopper(min_epoch=MIN_STOP)
optimizer = torch.optim.Adam(model.parameters())
epoch = 0
lowest_va_loss = 9999999999
best_ep = 0

def get_model_path(label):
    path = os.path.join(SAVE_FOLDER, f'{TRAINING_NAME}.{label}')
    return path

# load state for amp
if IS_CONTINUE:
    CONTINUE_PATH = get_model_path('last')
    print('continue on last epoch')
    print(CONTINUE_PATH)
    print()
    checkpoint = torch.load(CONTINUE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # amp.load_state_dict(checkpoint['amp_state_dict'])
    epoch = checkpoint['epoch']
    lowest_va_loss = checkpoint['lowest_va_loss']
    best_ep = checkpoint['best_ep']
    last_train_params = checkpoint['train_params']
    stopper = Stopper(epoch=epoch, best_loss=lowest_va_loss, min_epoch=MIN_STOP)
    print('loaded epoch ->', epoch)
    if NEW_LEARNING_RATE is not None:
        print('change learning rate to 10**', NEW_LEARNING_RATE)
        # scale learning rate
        update_per_epoch = len(training_set_loader)/BATCH_SIZE
        learning_rate = NEW_LEARNING_RATE/update_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
else:
    learning_rate = LEARNING_RATE


log_root = '/host' if os.path.exists('/host') else None
log = SKLogger(TRAINING_NAME, root=log_root)

loaded_path = CONTINUE_PATH if IS_CONTINUE else None
continue_ep = epoch if IS_CONTINUE else None

setting = ['*** Setting ***']
setting.append(f'PARAMS={params}')
setting.append(f'NAME={TRAINING_NAME}')
setting.append(f'BATCH={BATCH_SIZE}')
setting.append(f'TR:VA={len(training_set)}:{len(validation_set)}')
setting.append(f'LR={LEARNING_RATE:.0e}')
setting.append(f'NW={N_WORKERS}')
setting.append(f'DEVICE={DEVICE}')
setting.append(f'MODEL={model}')
setting.append(f'CONTINUE={IS_CONTINUE}')
setting.append(f'CHECKPOINT={loaded_path}')
setting.append(f'CONTINUE_EP={continue_ep}')
setting.append(f'LOG_DIR={log.dir}')
setting.append(f'MIN_STOP={MIN_STOP}')
setting = '\n'.join(setting)
print()
print(setting)
print()

def train(profile=False):
    global model, optimizer, epoch
    model.train()
    epoch += 1
    losses = []
    if profile:
        t0 = time.time()
    n = len(training_set_loader)
    last_iter = n-1 
    for iteration, dat in enumerate(training_set_loader):
        if iteration==last_iter and profile:
            t1 = time.time()
        loss = feed(dat)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if iteration==last_iter and profile:
            t2 = time.time()
    loss = avg(losses)
    if profile:
        print('n_iter=', n)
        print(t2-t1, 'one batch time')
        print(t2-t0, 'one ep time')

    if CHECKING:
        print('''
        ################################
        #                              #
        #                              #
        #    this is checking mode     #
        #                              #
        #                              #
        ################################
        ''')
        print('ep', epoch, '---loss- %.6f'%loss)
    return loss

def val_feed():
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

    loss = avg(losses)
    if CHECKING:
        print('ep', epoch, '-----------va- %.6f'%loss)
        time.sleep(0.3)
    return loss

def validation(tr_loss, profile=False):
    global lowest_va_loss, best_ep
    if profile:
        t0 = time.time()
    if epoch == 1 or epoch % SAVE_EVERY == 0:
        if profile:
            t1 = time.time()
        save_model('last')
        if profile:
            t2 = time.time()
        print('saved ep=', epoch)

        # validate if model is saved
        if profile:
            t3 = time.time()
        va_loss = val_feed()
        if profile:
            t4 = time.time()

        if va_loss < lowest_va_loss:
            # save best weight
            lowest_va_loss = va_loss
            best_ep = epoch
            save_model('best')
            print('*** saved best ep=', epoch)
    if profile:
        print(t4-t3, 'val_feed()')
        print(t2-t1, 'save_model()')
    write_loss(epoch, tr_loss, va_loss)
    return va_loss

def save_model(label):
    d = {
        'lowest_va_loss': lowest_va_loss,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_params': [str(args)],
        'setting': setting,
        'best_ep': best_ep,
    }
    if IS_CONTINUE:
        last_train_params.append(str(args))
        d['train_params'] = last_train_params
    path = get_model_path(label)
    # torch.save(d, path, _use_new_zipfile_serialization=False)
    # if error this line, check file permission
    torch.save(d, path)
    
def avg(losses: list):
    return sum(losses) / len(losses)

def main():
    global lowest_va_loss
    profile = args.profile
    while True:
        tr_loss = train(profile)
        va_loss = validation(tr_loss, profile)
        if CHECKING: break
        if profile: break
        if stopper(va_loss):
            print('breaked by stopper at ep', epoch)
            break
    print('done')

def write_loss(epoch, tr, va):
    log.write(epoch, tr, va)

if __name__ == '__main__':
    main()
    # python train.py name -ck
