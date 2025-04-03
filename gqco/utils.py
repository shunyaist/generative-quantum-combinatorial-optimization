import os
import sys
import random
import pickle

import numpy as np
import torch






def fix_seed(seed=373):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
    except:
        return None





def print0(text, args, terminal=False):
    if args.rank==0:
        log_file = open(f'{args.out_dir}/output.log', 'a')
        sys.stdout = log_file
        print(text)
        log_file.close()
        sys.stdout = sys.__stdout__

        if terminal:
            print(text)





def save_file(obj, file, args):
    if args.rank==0:
        with open(file, 'wb') as f:
            pickle.dump(obj, f)




def arange_token(token, args):
    while token and token[-1] == 0:
        token.pop()

    return token



def get_answer(dct, metric='min'):
    vals = list(dct.values())
    if metric=='min':
        target_val = min(vals)
    if metric=='max':
        target_val = max(vals)
    target_keys = [key for key, value in dct.items() if value == target_val]
    return target_keys, target_val