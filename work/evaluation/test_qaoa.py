import sys
print(sys.executable)

sys.path.append('../')


import pickle
import time
import tqdm
import numpy as np
import argparse
import cudaq
from typing import List
from gqco.quantum import make_cudaq_operator



parser = argparse.ArgumentParser(description='Megatron-LM Arguments', allow_abbrev=False)
group = parser.add_argument_group(title='setting')
group.add_argument('--layer-count', type=int, default=2, help='Layer count')
group.add_argument('--size', type=int, default=3, help='Number of qubits')
this_args = parser.parse_args()


layer_count = this_args.layer_count
size = this_args.size


def adj_to_cudaqlist(adj):
    lst1 = []
    lst2 = []
    
    for i in range(len(adj)):
        lst1.append(adj[i, i].cpu().tolist())
    for i in range(len(adj)-1):
        for j in range(i+1, len(adj)):
            lst2.append(adj[i, j].cpu().tolist())

    return lst1, lst2


@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, lst1: List[float],
                lst2: List[float], thetas: List[float]):

    qreg = cudaq.qvector(qubit_count)

    h(qreg)

    for ell in range(layer_count):

        for i in range(qubit_count):
            w = lst1[i]
            rz(2.0 * thetas[ell] * w, qreg[i])

        count = 0
        for i in range(qubit_count-1):
            for j in range(i+1, qubit_count):
                w = lst2[count]
                x.ctrl(qreg[i], qreg[j])
                rz(2.0 * thetas[ell] * w, qreg[j])
                x.ctrl(qreg[i], qreg[j])
                count += 1
            
        # Add the mixer kernel to each layer
        for k in range(qubit_count):
            rx(2.0 * thetas[ell + layer_count], qreg[k])


def _get_answer(dct, metric='min'):
    vals = list(dct.values())
    if metric=='min':
        target_val = min(vals)
    if metric=='max':
        target_val = max(vals)
    target_keys = [key for key, value in dct.items() if value == target_val]
    return target_keys, target_val



def qaoa_solver(adj, layer_count):
    # SEED = 0
    SEED = int(abs((adj[0,0].cpu().tolist() + adj[1,1].cpu().tolist())*10))

    start = time.time()

    lst1, lst2 = adj_to_cudaqlist(adj)
    spin_operator = make_cudaq_operator(adj)

    qubit_count = len(adj)
    parameter_count = 2 * layer_count
    
    cudaq.set_random_seed(SEED)
    optimizer = cudaq.optimizers.NelderMead()
    # optimizer = cudaq.optimizers.COBYLA()
    np.random.seed(SEED)
    optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, parameter_count)
    # optimizer.initial_parameters = np.random.uniform(-np.pi, np.pi, parameter_count)
    optimizer.max_iterations = 1000

    def objective(parameters):
        return cudaq.observe(kernel_qaoa, spin_operator, qubit_count, layer_count, lst1, lst2, parameters).expectation()
    
    optimal_expectation, optimal_parameters = optimizer.optimize(dimensions=parameter_count, function=objective)

    np.array(cudaq.get_state(kernel_qaoa, qubit_count, layer_count, lst1, lst2, optimal_parameters))  ## the output is wrong without this line (why?).
    vector = np.array(cudaq.get_state(kernel_qaoa, qubit_count, layer_count, lst1, lst2, optimal_parameters)) 
    probs = np.abs(vector)**2

    result = {}
    nqubit = int(np.log2(len(probs)))
        
    for i, b in enumerate(range(2**nqubit)):
        # bit = f'{b:0>{nqubit}b}'
        bit = f'{b:0>{nqubit}b}'[::-1]
        result[bit] = probs[i]

    min_keys, min_val = _get_answer(result, metric='max')

    exe_time = time.time() - start

    return min_keys, exe_time




testdata_path = './testdata.pkl'
with open(testdata_path, 'rb') as f:
    testdata = pickle.load(f)




ans = {}
tms = {}
_ans = []
_tms = []

count = 0
for adj in tqdm.tqdm(testdata[size], desc=f'layer_count: {layer_count}, size: {size}'):
    count += 1
    min_keys, exe_time = qaoa_solver(adj, layer_count)

    _ans.append(min_keys)
    _tms.append(exe_time)

    # if count == 10:
    #     break
    
with open(f'./outputs/qaoaans_l{layer_count}_s{size}.pkl', 'wb') as f:
    pickle.dump({
        'answer': _ans,
        'time': _tms
    }, f)

print('end')