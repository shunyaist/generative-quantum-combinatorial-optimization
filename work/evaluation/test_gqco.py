import sys
print(sys.executable)

sys.path.append('../')


import pickle
import time
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse
from torch_geometric.data import Batch

from gqco.train import MyModel
from gqco.utils import fix_seed, arange_token
from gqco.data import generate_data, RandomGraphDatasetWithClone
from gqco.solve import solve_from_token, plot_from_dict, brute_solver, probs_to_result
from gqco.model import TransformerWithMoE
from gqco.quantum import coef_to_pauli, pauli_to_matrix
from gqco.task import GQCO



parser = argparse.ArgumentParser(description='Megatron-LM Arguments', allow_abbrev=False)
group = parser.add_argument_group(title='setting')
group.add_argument('--temperature', type=float, default=1.0, help='temperature')
group.add_argument('--num-clone', type=int, default=10, help='Number of sampling')
group.add_argument('--size', type=int, default=3, help='Number of qubits')
group.add_argument('--seed', type=int, default=373, help='seed')
this_args = parser.parse_args()




def data_from_adj(adj, args, num_clone, device):
    dataset = RandomGraphDatasetWithClone(adj, num_clone=num_clone, max_num_nodes=args.max_size, device=device)
    dataset.x = dataset.x.half()
    dataset.edge_attr = dataset.edge_attr.half()
    record = Batch.from_data_list(dataset)
    record['size'] = record['size'].tolist()[0]
    record['len'] = dataset.len()

    return adj, size, record


def _get_answer(dct, metric='min'):
    vals = list(dct.values())
    if metric=='min':
        target_val = min(vals)
    if metric=='max':
        target_val = max(vals)
    target_keys = [key for key, value in dct.items() if value == target_val]
    return target_keys, target_val



task_path = '../model/taskobjects.pkl'
checkpoint_path = '../model/merged_model.ckpt'
testdata_path = './testdata.pkl'
seed=0
SEED = this_args.seed
num_clone = this_args.num_clone
temperature = this_args.temperature
size = this_args.size


with open(task_path, 'rb') as f: 
    obj = pickle.load(f)
args = obj['args']
task = GQCO(args)
task.tool = 'cudaq'

model = TransformerWithMoE(args)
model = MyModel.load_from_checkpoint(checkpoint_path, model=model, task=task, args=args)
model = model.to('cuda')
device = model.device
model = model.model




with torch.no_grad():
    for s in range(3, args.max_size+1):
        if s != size:
            del model.encoder.ffns_in_e[f'Expert-{s}']
            del model.encoder.ffns_in[f'Expert-{s}']
            for j in range(len(model.encoder.convolution.convs)):
                del model.encoder.convolution.convs[j].ffns[f'Expert-{s}']
            for j in range(len(model.decoder.layers)):
                del model.decoder.layers[j].ffns[f'Expert-{s}']
            del model.lms[f'Expert-{s}']

        # Remove corresponding parameters from the state_dict
        state_dict = model.state_dict()
        keys_to_remove = [key for key in state_dict.keys() if f'Expert-{s}' in key]
        for key in keys_to_remove:
            del state_dict[key]

        # Load the modified state_dict back into the model (optional)
        model.load_state_dict(state_dict, strict=False)

model_comp = torch.compile(model, mode='max-autotune')



with open(testdata_path, 'rb') as f:
    testdata = pickle.load(f)
with open('./outputs/trueans.pkl', 'rb') as f:
    dict_true = pickle.load(f)



size_list = [s for s in range(3, 11)]



model_comp.eval()
ans = {}
tms = {}
_ans = []
_tms = []
_tokens = []

print(device)
import cudaq
print(cudaq.get_target().name)


adj, size, record = data_from_adj(testdata[size][0].to(device), args, num_clone, device)
for tt in range(20):
    adj, size, record = data_from_adj(testdata[size][tt].to(device), args, num_clone, device)
    with torch.no_grad():
        with torch.autocast('cuda'):
            _ = model_comp.forward(record, temperature=temperature, same_token_penalty=0.0, masked_tokens=task.bad_tokens[size], deterministic=False) 

count = 0
for itr in tqdm.tqdm(range(len(testdata[size])), desc=f'temperature: {temperature}, clone: {num_clone}, size: {size}'):
    adj = testdata[size][itr]
    count += 1
    adj, size, record = data_from_adj(adj.to(device), args, num_clone, device)
    fix_seed(SEED)

    # with torch.no_grad():
    with torch.no_grad():
        with torch.autocast('cuda'):
            _s = time.time()
            out_tokens, probs_all, _, logits_all = model_comp.forward(record, temperature=temperature, same_token_penalty=0.0, masked_tokens=task.bad_tokens[size], deterministic=False) 
            _t1 = time.time()
    tokens_list = [arange_token(t, args) for t in out_tokens.detach().cpu().tolist()]

    is_correct = 0
    t_cache = []
    t_best = None
    for t in tokens_list:
        t_tuple = tuple(t)
        if t_tuple not in t_cache:
            t_cache.append(t_tuple)

            trueans = dict_true['answer'][size][itr]

            while t and t[-1] == 0:
                t.pop()

            qc = task.get_circuit(t, size=len(adj))
            vector = qc.get_state()
            probs = np.abs(vector)**2

            dict_pred = probs_to_result(probs)
            min_keys, min_val = _get_answer(dict_pred, metric='max')

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

        
with open(f'./outputs/gqcoans_s{SEED}_t{temperature}_cl{num_clone}_s{size}.pkl', 'wb') as f:
    pickle.dump({
        'answer': _ans,
        'time': _tms,
        'tokens': _tokens
    }, f)

print('end')