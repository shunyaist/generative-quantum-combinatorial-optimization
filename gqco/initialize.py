import wandb
from datetime import datetime

from .arguments import parse_args, print_args






def initialize_all():

    ## Set arguments
    args = parse_args()

    ## Time
    args.start_date = current_time = datetime.now().strftime('%Y/%m/%d/%H:%M:%S')

    ## Log file
    if args.rank==0:
        with open(f'{args.out_dir}/output.log', 'w') as f:
            f.write(f'Start time: {args.start_date}')

    print_args('arguments', args)

    return args


