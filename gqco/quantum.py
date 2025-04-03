
import numpy as np
from math import pi
import torch

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXGate, RYGate, RZGate, PhaseGate, CXGate, IGate, CRXGate, TGate, HGate, SdgGate, TdgGate, SGate, XGate, RZZGate
import cudaq
from cudaq import spin
from IPython.display import display








class Circuit():
    def __init__(self, tool, num_qubit):
        self.tool = tool
        self.num_qubit = num_qubit

        if self.tool == 'qiskit':
            self.qc = QuantumCircuit(self.num_qubit)

        if self.tool == 'cudaq':
            self.kernel = cudaq.make_kernel()
            self.qreg = self.kernel.qalloc(self.num_qubit)




    def add_gate(self, operators):
        for opr in operators:

            if self.tool == 'qiskit':
                g = self.str_to_gate(opr['type'], theta=opr['theta'])
                self.qc.append(g, opr['control']+opr['loc'])

            if self.tool == 'cudaq':
                self.append(opr)



    def str_to_gate(self, _type, theta=None):
        ## For qiskit
        if _type=='X':
            return XGate()
        if _type=='RX':
            return RXGate(theta)
        if _type=='RY':
            return RYGate(theta)
        if _type=='RZ':
            return RZGate(theta)
        if _type=='P':
            return PhaseGate(theta)
        if _type=='CX':
            return CXGate()
        if _type=='CRX':
            return CRXGate(theta)
        if _type=='T':
            return TGate()
        if _type=='t':
            return TdgGate()
        if _type=='S':
            return SGate()
        if _type=='s':
            return SdgGate()
        if _type=='H':
            return HGate()
        if _type=='I':
            return IGate()
        if _type=='RZZ':
            return RZZGate(theta)
        raise AssertionError(f'The gate {_type} is not available.')



    def append(self, opr):
        ## For cudaq
        _type = opr['type']
        _loc = opr['loc'][0]
        if _type=='X':
            self.kernel.x(self.qreg[_loc])
        elif _type=='RX':
            self.kernel.rx(opr['theta'], self.qreg[_loc])
        elif _type=='RY':
            self.kernel.ry(opr['theta'], self.qreg[_loc])
        elif _type=='RZ':
            self.kernel.rz(opr['theta'], self.qreg[_loc])
        elif _type=='P':
            self.kernel.r1(opr['theta'], self.qreg[_loc])
        elif _type=='CX':
            self.kernel.cx(self.qreg[opr['control'][0]], self.qreg[_loc])    ## TODO code for multi-control-NOT
        elif _type=='CRX':
            self.kernel.crx(opr['theta'], self.qreg[opr['control'][0]], self.qreg[_loc])
        elif _type=='T':
            self.kernel.t(self.qreg[_loc])
        elif _type=='t':
            self.kernel.r1(-pi/4, self.qreg[_loc])
        elif _type=='S':
            self.kernel.s(self.qreg[_loc])
        elif _type=='s':
            self.kernel.r1(-pi/2, self.qreg[_loc])
        elif _type=='H':
            self.kernel.h(self.qreg[_loc])
        elif _type=='I':
            self.kernel.rx(0.0, self.qreg[_loc])
        elif _type=='RZZ':
            self.kernel.cx(self.qreg[opr['loc'][0]], self.qreg[opr['loc'][1]])
            self.kernel.rz(opr['theta'], self.qreg[opr['loc'][1]])
            self.kernel.cx(self.qreg[opr['loc'][0]], self.qreg[opr['loc'][1]])
        else:
            raise AssertionError(f'The gate {_type} is not available.')



    def get_state(self):
        if self.tool == 'qiskit':
            transpiled = transpile(self.qc.decompose(), optimization_level=1)
            vector = Statevector(transpiled)
            array = []
            for v in vector:
                array.append(v)
            return np.array(array)

        if self.tool == 'cudaq':
            np.array(cudaq.get_state(self.kernel))  ## the output is sometimes wrong without this line (TODO why?).
            return np.array(cudaq.get_state(self.kernel))
        


    def expectation(self, adj, num_shot=-1):

        paulis, coeffs = coef_to_pauli(adj)

        if num_shot > 0:
            if self.tool == 'cudaq':
                spin_operator = make_cudaq_operator(adj)
                energy = cudaq.observe(self.kernel, spin_operator, shots_count=num_shot).expectation()

        else:
            if self.tool == 'qiskit':
                H = np.diag(np.zeros(pow(2, self.num_qubit), dtype=np.complex128))
                for p, h in zip(paulis, coeffs):
                    H += h * pauli_to_matrix(p)

                vector = self.get_state()
                energy = np.vdot(vector, H @ vector).real

            if self.tool == 'cudaq':
                spin_operator = make_cudaq_operator(adj)
                energy = cudaq.observe(self.kernel, spin_operator, shots_count=num_shot).expectation()

            
        return energy



    def draw(self, output='mpl', style='bw', scale=0.35, savefile=None, is_transpile=False):

        if self.tool == 'cudaq':
            np.array(cudaq.get_state(self.kernel))  ## the output is sometimes wrong without this line (TODO why?).
            print(cudaq.draw(self.kernel))

            ## TODO
            ## transpile code for cudaq
            ##

        elif self.tool == 'qiskit':

            if is_transpile:
                transpiled_qc = transpile(self.qc.decompose(), optimization_level=1)
                display(transpiled_qc.draw(output=output, style=style, scale=scale))
                if savefile is not None:
                    transpiled_qc.draw(output=output, style=style, scale=scale, filename=f'{savefile}-transqc.svg')

            else:
                display(self.qc.draw(output=output, style=style, scale=scale))
                if savefile is not None:
                    self.qc.draw(output=output, style=style, scale=scale, filename=f'{savefile}-qc.svg')






def zfill_adj(adj, fill_size):

    adj_size = len(adj)

    out = torch.zeros((fill_size, fill_size))
    for i in range(adj_size):
        for j in range(i, adj_size):
            out[i, j] = adj[i, j].detach()
    
    return out









class Pauli:
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])








def pauli_to_matrix(p):
    m = {"X": Pauli.X, "Y": Pauli.Y,
        "Z": Pauli.Z, "I": Pauli.I}
    matrix = None
    for c in p:
        if matrix is None:
            matrix = m[c]
        else:
            matrix = np.kron(m[c], matrix)
    return matrix








def MakePauli(i, j, n, op='X'):
    s = 'I'*n
    s_list = list(s)
    s_list[i] = op
    s_list[j] = op
    return ''.join(s_list)








def coef_to_pauli(adj):
    size = len(adj)

    paulis = []
    coeffs = []

    for i in range(size):
        for j in range(i, size):
            w = adj[i, j].detach().tolist()
            if w != 0:
                paulis.append(MakePauli(i, j, size, 'Z'))
                coeffs.append(w)

    return paulis, coeffs








def make_cudaq_operator(adj):
    size = len(adj)

    opr = 0
    for i in range(size):
        for j in range(i, size):
            w = adj[i, j].detach().tolist()
            if w != 0:
                if i == j:
                    opr += w * spin.z(i)
                else:
                    opr += w * spin.z(i) * spin.z(j)

    return opr            
