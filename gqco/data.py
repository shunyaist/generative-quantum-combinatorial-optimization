
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, Batch

from gqco.utils import fix_seed








def make_feat(w, adj, loc=[]):
    k = loc[0]
    ell = loc[1]
    l = len(adj)
    feat = [w]
    for i in range(l):
        for j in range(i, l):

            if k==ell:  ## if source is node
                if i==j:  ## if target is node

                    diff = w - adj[i, j]
                    diff = torch.sign(diff)
                    feat += [diff]


                    if k < i:
                        out1 = w - adj[k, i]
                        out2 = torch.sign(w*adj[i, j]*adj[k, i])
                    if k > i:
                        out1 = w - adj[i, k]
                        out2 = torch.sign(w*adj[i, j]*adj[i, k])
                    if k == i:  ## if target is the same as source
                        out1 = torch.sign(w**2)*0   ## = 0
                        out2 = torch.sign(w**2)*0  ## = 0
                    feat += [torch.sign(out1)]
                    feat += [out2]

            if k!=ell:  ## if source is edge
                if i==k and j==ell:  ## if target is corresponding node

                    diff1 = torch.sign(w - adj[k, k])
                    diff2 = torch.sign(w - adj[ell, ell])
                    feat += [diff1]
                    feat += [diff2]

                    out = torch.sign(w*adj[k, k]*adj[ell, ell])
                    feat += [out]

    return torch.stack(feat)


def compute_node_dim(size=None):
    return int(1+size*3)


def compute_edge_dim(size=None):
    return int(4)





class RandomGraphDatasetWithClone(InMemoryDataset):

    def __init__(self, adj, num_clone, device):
        super().__init__('.', None)
        self.num_clone = num_clone
        self.adj = adj
        self.device = device
        self.data, self.slices = self.process_data()


    def process_data(self):

        size = len(self.adj)

        edge_indx = []
        edge_feat = []
        node_feat = []

        for i in range(size):
            for j in range(i, size):
                w = self.adj[i, j]
                w_feat = make_feat(w, self.adj, loc=[i, j])

                if i==j:
                    node_feat += [w_feat]
                else:
                    edge_feat += [w_feat]
                    edge_indx += [[i, j]]
                    edge_feat += [w_feat]
                    edge_indx += [[j, i]]


        self.dim_feat = len(w_feat)

        data_list = []
        for _ in range(self.num_clone):
            data = Data(
                x = torch.stack(node_feat),
                edge_index = torch.tensor(edge_indx, dtype=torch.long, device=self.device).t().contiguous(),
                edge_attr = torch.stack(edge_feat)
            )

            data.size = len(self.adj)
            data_list.append(data)

        return self.collate(data_list)
    




class DummyDataset(Dataset):
    def __init__(self):
        self.dummy = 1

    def __len__(self):
        return self.dummy

    def __getitem__(self, idx):
        x = torch.randn(self.dummy)  # Random input
        return x
    





    

def generate_data(args, seed, device, num_clone=-1, size=None, current_size=None):

    fix_seed(seed)
    min_size = 3

    ## set problem size
    if args.tune_size > 0:
        size = args.tune_size

    if (args.tune_size < 0) & (size is None) & (current_size is not None):

        if current_size == min_size:
            size = current_size
        else:
            size_probs = np.array([0.5/(current_size-min_size) for _ in range(args.max_size - min_size + 1)])
            size_probs[current_size - min_size] = 0.5
            size_probs[(current_size - min_size + 1):] = 0
            size = random.choices([s for s in range(min_size, args.max_size+1)], size_probs.tolist())[0]

    ## generate coefficient matrix
    adj = generate_adj(size, seed=seed, device=device)

    if num_clone < 0:
        clone_size = {
            '3': 1024,
            '4': 1024,
            '5': 512,
            '6': 384,
            '7': 256,
            '8': 192,
            '9': 128,
            '10': 96
        }
        num_clone = clone_size[str(size)]


    ## create copied dataset
    adj, _, record = data_from_adj(adj, num_clone, device)

    return adj, size, record






def generate_adj(size, seed=0, device='cpu'):

    fix_seed(seed)
    
    ## generate coefficient matrix
    adj = torch.zeros((size, size), device=device)
    for i in range(size):
        for j in range(i, size):
            adj[i,j] = torch.rand(1, device=device)*2-1   ## in [-1, 1]
    adj = adj / torch.max(torch.abs(adj))

    return adj





def data_from_adj(adj, num_clone, device):
    dataset = RandomGraphDatasetWithClone(adj, num_clone=num_clone, device=device)
    record = Batch.from_data_list(dataset)
    record['size'] = record['size'].tolist()[0]
    record['len'] = dataset.len()

    return adj, record['size'], record
