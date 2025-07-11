import concurrent.futures
import sys
sys.path.append('../../')

from gqco.utils import fix_seed, get_answer
import networkx as nx
import torch
import numpy as np
import pickle
import time
import tqdm
from gqco.solve import brute_solver

seed = 373
num_problems = 1000
size = 10


def ragular_mask(size, degree=3, seed=0, device='cpu'):
    fix_seed(seed)
    G = nx.random_regular_graph(degree, size, seed=seed)
    msk_matrix = nx.to_numpy_array(G)
    msk = torch.triu(torch.from_numpy(msk_matrix)).float().to(device)
    return msk


def generate_adj(size, seed=0, device='cpu'):
    fix_seed(seed)
    adj = torch.zeros((size, size), device=device)
    for i in range(size):
        for j in range(i, size):
            adj[i, j] = torch.rand(1, device=device)*2 - 1
    adj = adj / torch.max(torch.abs(adj))
    return adj


def solve_degree_problems(degree):
    tmp_adjs, _ans, _tms = [], [], []
    for seed in range(num_problems):
        adj = generate_adj(size, seed=seed*100 + degree)
        msk = ragular_mask(size, degree=degree, seed=seed*100 + degree)
        adj = adj * msk
        tmp_adjs.append(adj)

    for adj in tqdm.tqdm(tmp_adjs, desc=f'degree: {degree}'):
        start_time = time.time()
        dict_true = brute_solver(adj)
        end_time = time.time()

        min_keys_true, min_val_true = get_answer(dict_true, metric='min')
        _ans.append(min_keys_true)
        _tms.append(end_time - start_time)

    return degree, tmp_adjs, _ans, _tms


adjs, ans, tms = {}, {}, {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(solve_degree_problems, degree): degree for degree in range(3, 10)}
    for future in concurrent.futures.as_completed(futures):
        degree, tmp_adjs, _ans, _tms = future.result()
        adjs[degree] = tmp_adjs
        ans[degree] = _ans
        tms[degree] = _tms


with open('../../data/testdata_10regular.pkl', 'wb') as f:
    pickle.dump(adjs, f)

with open('../../data/trueans_10regular.pkl', 'wb') as f:
    pickle.dump({'answer': ans, 'time': tms}, f)
