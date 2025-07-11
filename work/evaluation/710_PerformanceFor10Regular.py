import sys

sys.path.append('./')
print(sys.path)


import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import time
import os
import torch
from IPython.display import display, Math
from gqco.train import MyModel
from gqco.utils import fix_seed, arange_token, get_answer
from gqco.data import generate_adj, data_from_adj
from gqco.solve import solve_from_token, plot_from_dict, adj_to_text, brute_solver, probs_to_result
from gqco.model import TransformerWithMoE, PositionalEmbedding, param_clone

SEED = 42



def prepare_model(do_copy=False, ckpt=None):
    task_path = 'model/taskobjects.pkl'

    if ckpt is None:
        checkpoint_path = 'model/model_10regular.ckpt'
    else:
        checkpoint_path = ckpt

    print(f'Loading model from {checkpoint_path}...')

    with open(task_path, 'rb') as f: 
        obj = pickle.load(f)
    gqco = obj['task']
    args = obj['args']

    model = TransformerWithMoE(args)
    model = MyModel.load_from_checkpoint(checkpoint_path, model=model, task=gqco, args=args)
    
    try:
        model = model.to('cuda')
        device = model.device
    except:
        device = torch.device('cpu')

    gqco.tool = 'cudaq'
    model = model.model
    model.eval()

    return model, gqco, args, device






def performance_eval(args, task, model, testdata, degree, size, dict_true, num_clone=100, device='cpu', flg=''):
    _ans = []
    _tms = []
    _tokens = []

    temperature = 2.0
    count = 0

    savepath = f'./work/evaluation/outputs/{flg}gqcoans_s{SEED}_t{temperature}_cl{num_clone}_s{size}_degree{degree}.pkl'
    print(f'Saving results to {savepath}...')    


    for itr in tqdm.tqdm(range(len(testdata[degree])), desc=f'temperature: {temperature}, clone: {num_clone}, size: {size}, degree: {degree}, flg: {flg}'):
        adj = testdata[degree][itr]
        count += 1
        adj, size, record = data_from_adj(adj.to(device), num_clone, device)
        fix_seed(SEED)

        model.to(device)
        # with torch.no_grad():
        with torch.no_grad():
            with torch.autocast('cuda'):
                _s = time.time()
                out_tokens, probs_all, _, logits_all = model.forward(record, temperature=temperature, masked_tokens=task.bad_tokens[size]) 
                _t1 = time.time()
        tokens_list = [arange_token(t, args) for t in out_tokens.detach().cpu().tolist()]

        is_correct = 0
        t_cache = []
        t_best = None
        for t in tokens_list:
            t_tuple = tuple(t)
            if t_tuple not in t_cache:
                t_cache.append(t_tuple)

                trueans = dict_true['answer'][degree][itr]

                while t and t[-1] == 0:
                    t.pop()
                qc = task.get_circuit(t, size=len(adj))
                vector = qc.get_state()
                probs = np.abs(vector)**2

                dict_pred = probs_to_result(probs)
                min_keys, min_val = get_answer(dict_pred, metric='max')

                if len(set(min_keys) & set(trueans)):
                    is_correct += 1
                    t_best = t
                    break

        _t2 = time.time()

        if t_best is None:
            min_energy = 10000
            t_cache2 = []
            for tt in tokens_list:
                t_tuple = tuple(tt)
                if t_tuple not in t_cache2:
                    energy = task.compute_energy(tt, adj, args.num_shot)

                    if energy < min_energy:
                        t_best = tt


        _ans.append(is_correct)
        _tms.append([_s, _t1, _t2])
        _tokens.append(t_best)

    savepath = f'./work/evaluation/outputs/{flg}gqcoans_s{SEED}_t{temperature}_cl{num_clone}_s{size}_degree{degree}.pkl'

    with open(savepath, 'wb') as f:
        pickle.dump({
            'answer': _ans,
            'time': _tms,
            'tokens': _tokens
        }, f)

    print('end')







def main(kickid):

    size = 10

    ## data preparation
    with open('data/testdata_10regular.pkl', 'rb') as f:
        testdata = pickle.load(f)
    with open('data/trueans_10regular.pkl', 'rb') as f:
        dict_true = pickle.load(f)


    pid = '743377'
    epoch = 1399

    checkpoint_path = f'./model/model_finetuned.ckpt'
    model, gqco, args, device = prepare_model(do_copy=False, ckpt=checkpoint_path)
    performance_eval(args, gqco, model, testdata, 3, size, dict_true, num_clone=100, device=device, flg=f'finetuned_')





    


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='setting')
    group.add_argument('--kick-id', type=int, default=0, help='kickid')
    args = parser.parse_args()

    fix_seed(SEED)
    main(args.kick_id)
    print('Done.')
