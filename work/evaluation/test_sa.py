import sys
print(sys.executable)

sys.path.append('../')


import pickle
import time
import tqdm
import numpy as np
import argparse

import neal


parser = argparse.ArgumentParser(description='Megatron-LM Arguments', allow_abbrev=False)
group = parser.add_argument_group(title='setting')
group.add_argument('--num-reads', type=int, default=10, help='NUmber of reads')
group.add_argument('--num-sweeps', type=int, default=10, help='Number of sweeps')
group.add_argument('--size', type=int, default=3, help='Number of qubits')
this_args = parser.parse_args()


num_reads = this_args.num_reads
num_sweeps = this_args.num_sweeps
size = this_args.size


def sa_solver(adj):
    start = time.time()
    
    np_adj = np.array(adj.tolist())
    
    linear = {i: np_adj[i, i] for i in range(len(np_adj))}
    
    quadratic = {(i, j): np_adj[i, j] for i in range(len(np_adj)) for j in range(i + 1, len(np_adj)) if np_adj[i, j] != 0}
    
    sampler = neal.SimulatedAnnealingSampler()
    
    response = sampler.sample_ising(linear, quadratic, num_reads=num_reads, num_sweeps=num_sweeps)
    
    ans_min = None
    ene_min = 100
    for rec in response.record:
        ans, ene, _ = rec
        if ene < ene_min:
            ans_min = ans
    
    ans_bit = (1 - ans_min)/2
    result = [''.join(ans_bit.astype(int).astype(str))]

    exe_time = time.time() - start

    return result, exe_time




testdata_path = './testdata.pkl'
with open(testdata_path, 'rb') as f:
    testdata = pickle.load(f)




ans = {}
tms = {}
_ans = []
_tms = []

count = 0
for adj in tqdm.tqdm(testdata[size], desc=f'num_reads: {num_reads}, sweeps: {num_sweeps}, size: {size}'):

    min_keys, exe_time = sa_solver(adj)

    _ans.append(min_keys)
    _tms.append(exe_time)

    if count == 10:
        break
    
with open(f'./outputs/saans_r{num_reads}_sw{num_sweeps}_s{size}.pkl', 'wb') as f:
    pickle.dump({
        'answer': _ans,
        'time': _tms
    }, f)

print('end')