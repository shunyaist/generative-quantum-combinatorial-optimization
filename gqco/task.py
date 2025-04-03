
from math import pi
import numpy as np

from .quantum import Circuit





def define_tokens(num_qubit, rot=1, rzz=1, had=1, cnot=1):
    
    ## Initialize
    pool = {}
    opr = []

    ## Identity
    opr.append({'type': 'I', 'loc': [0], 'control': [], 'theta': None})
    pool['0'] = opr
    idx = 1
    identity_id = 0

    ## Rotation gate
    if True: #rot > 0:
        for _type in ['RX', 'RY', 'RZ']:
            for theta in [sign*angle for sign in [1, -1] for angle in [pi/3, pi/4, pi/5]]:
                for pos in range(num_qubit):
                    opr = []
                    opr.append({'type': _type,  'loc': [pos], 'control': [], 'theta': theta})
                    pool[str(idx)] = opr
                    idx += 1

    ## 2-qubit Rotation gate
    if True: #rzz > 0:
        for _type in ['RZZ']:
            for theta in [sign*angle for sign in [1, -1] for angle in [pi/3, pi/4, pi/5]]:
                for i in range(num_qubit-1):
                    for j in range(i+1, num_qubit):
                        opr = []
                        opr.append({'type': _type, 'loc': [i, j], 'control': [], 'theta': theta})
                        pool[str(idx)] = opr
                        idx += 1

    ## Hadamard gate
    if True: #had > 0:
        for pos in range(num_qubit):
            opr = []
            opr.append({'type': 'H',  'loc': [pos], 'control': [], 'theta': None})
            pool[str(idx)] = opr
            idx += 1

    ## CNOT gate
    if True: #cnot > 0:
        for i in range(num_qubit):
            for j in range(num_qubit):
                if i != j:
                    opr = []
                    opr.append({'type': 'CX', 'loc': [i], 'control': [j], 'theta': None})
                    pool[str(idx)] = opr
                    idx += 1

    ## Specify bad tokens
    max_pos = {}
    bad_tokens = {}
    for opr_key in pool.keys():
        pos = []
        for opr in pool[opr_key]:
            _loc = opr['loc']
            _control = opr['control']
            pos += _loc + _control
        max_pos[opr_key] = max(pos)
    for nq in range(1, num_qubit+1):
        bad_tokens[nq] = [int(key) for key in max_pos.keys() if max_pos[key]>(nq-1)]


    return pool, bad_tokens, identity_id







class GQCO():
    
    def __init__(self, args):

        self.tool = args.quantum_tool
        self.num_qubit = args.num_qubit
        self.pool, self.bad_tokens, self.identity_id = define_tokens(self.num_qubit)



    def get_circuit(self, token, size=None):
        '''
            get circuit from token sequence
        '''
        if size is None:
            myqc = Circuit(self.tool, self.num_qubit)
        else:
            myqc = Circuit(self.tool, size)

        if self.tool=='qiskit':
            for t in token:
                myqc.add_gate(self.pool[str(t)])

        return myqc



    def make_taskargs(self, args):
        
        args.vocab_size = len(self.pool)
        args.eos_token_id = self.identity_id
        args.pool = self.pool
        
        return args



    def compute_energy(self, token, adj, num_shot=-1):
        '''
            compute expectation from token and coefficient matrix
        '''
        while token and token[-1] == 0:
            token.pop()

        qc = self.get_circuit(token, size=len(adj))
        energy = qc.expectation(adj, num_shot)

        return energy
