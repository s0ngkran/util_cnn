import os
import sys

def read_all_fnames(folder):
    files = os.listdir(folder)
    return files

def ck_ep():
    saves = read_all_fnames("save")
    ress = read_all_fnames("result")
    ress = [x.replace(".res", "") for x in ress]
    saves = [x for x in saves if 'last' not in x]
    saves = sorted(saves)

    for s in saves:
        if s not in ress:
            print(s)

def ck_best():
    saves = read_all_fnames("save")
    saves = [x for x in saves if x.endswith(".best")]
    saves = [x for x in saves if 'last' not in x]
    saves = sorted(saves)
    ress = read_all_fnames("result")
    ress = [x.replace(".res", "") for x in ress if '.best' in x]

    for s in saves:
        if s not in ress:
            print(s)

def main(is_only_best=False):
    if is_only_best:
        ck_best() 
    else:
        ck_ep()

if __name__ == "__main__":
    first_arg = sys.argv[1]
    is_only_best = True if first_arg == 'best' else False
    main(is_only_best)
    # python read_avai.py best
    # python read_avai.py nobest
