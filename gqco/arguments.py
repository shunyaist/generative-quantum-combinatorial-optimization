import os
import time
import argparse

from .utils import print0





def parse_args():
    '''
        Parse all arguments
    '''
    parser = argparse.ArgumentParser(description='GQCO Arguments', allow_abbrev=False)

    ## Job setting
    group = parser.add_argument_group(title='job')
    group.add_argument('--job-id', type=int, default=10, help='ABCI job ID.')
    group.add_argument('--num-hosts', type=int, default=1, help='Number of hosts, i.e., GPU nodes.')
    group.add_argument('--out-dir', type=str, default='./', help='Outputs directory path.')
    group.add_argument('--seed', type=int, default=373, help='Random seed used for python, numpy, pytorch, and cuda.')
    group.add_argument('--is-wandb', action='store_true', help='If True, use wandb.')
    group.add_argument('--no-wandb', dest='is_wandb', action='store_false', help='If False, do not use wandb.')
    group.add_argument('--project-name', type=str, default='MyProject', help='WandB project name.')
    group.add_argument('--task-name', type=str, default=None, help='Task name.')
    group.add_argument('--task-type', type=str, default=None, help='Task type.')
    group.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    group.add_argument('--start-size', type=int, default=3, help='Problem size for starting')
    group.add_argument('--tune-size', type=int, default=-1, help='Expert tuning size. If negative, train all parameters')

    ## Quantum
    group = parser.add_argument_group(title='quantum')
    group.add_argument('--quantum-tool', type=str, default='qiskit', help='Tool for quantum computation.')
    group.add_argument('--num-qubit', type=int, default=None, help='Number of max qubit.')

    ## Model
    group = parser.add_argument_group(title='model')
    group.add_argument('--min-generation', type=int, default=5, help='Minimum token generation length.')
    group.add_argument('--max-generation', type=int, default=10, help='Maximum token generation length.')
    group.add_argument('--max-size', type=int, default=20, help='Maximum number of qubit of quantum circuit.')
    group.add_argument('--hidden-dim', type=int, default=64, help='Dimension of hidden layer.')

    group.add_argument('--encoder-type', type=str, default='Transformer', help='Encoder convolution type.')
    group.add_argument('--encoder-depth', type=int, default=8, help='Depth of encoder attention loops.')
    group.add_argument('--encoder-num-heads', type=int, default=8, help='Number of decoder multi attention head.')
    group.add_argument('--encoder-hidden-dim-inner', type=int, default=64, help='Dimansion of inner hidden layer of encoder.')

    group.add_argument('--decoder-num-heads', type=int, default=8, help='Number of decoder multi attention head.')
    group.add_argument('--decoder-depth', type=int, default=8, help='Deapth of encoder.')
    group.add_argument('--decoder-ffn-dim', type=int, default=64, help='Dimensiton of intermediate linear layer of encoder.')

    ## Training
    group = parser.add_argument_group(title='training')
    group.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    group.add_argument('--init-checkpoint', type=str, default=None, help='Initial checkpoint.')
    group.add_argument('--max-epoch', type=int, default=100000, help='Maximum Number of epoch.')
    group.add_argument('--ipo-beta', type=float, default=1.0, help='Beta parameter for ipo loss.')
    group.add_argument('--dpo-beta', type=float, default=1.0, help='Beta parameter for dpo loss.')
    group.add_argument('--num-policy-search', type=int, default=10, help='Number of policy search')
    group.add_argument('--num-shot', type=int, default=-1, help='Number of shots for quantum sampling. If you set negative value, compute exact value without sampling')
    group.add_argument('--log-freq', type=int, default=1, help='Frequency for logging.')
    group.add_argument('--log-freq-acc', type=int, default=1, help='Frequency for logging accuracy.')
    group.add_argument('--loss-mode', type=str, default='all-to-all', help='How to compute losses for every step.')
    group.add_argument('--loss-type', type=str, default='ipo', help='What type of loss to be used.')
    group.add_argument('--checkpoint-freq', type=int, default=10, help='Frequency for checkpointing.')

    ## tmp
    group = parser.add_argument_group(title='Gateset type')
    group.add_argument('--rot', type=int, default=1, help='Use 1-qubit rotation gates.')
    group.add_argument('--rzz', type=int, default=1, help='Use rzz gate.')
    group.add_argument('--had', type=int, default=1, help='Use hadamard gate.')
    group.add_argument('--cnot', type=int, default=1, help='Use cnot gate')
    group.add_argument('--gateset-name', type=str, default="", help='Gateset name.')


    args = parser.parse_args()

    args.start_time = time.time()
    args.rank = int(os.getenv('RANK', '0'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))


    return args








def print_args(title, args):
    '''
        Print arguments.
    '''
    print0('\n', args)
    print0(f'------------------------ {title} ------------------------', args)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print0(arg, args)
    print0(f'-------------------- end of {title} ---------------------', args)
    print0('\n', args)
