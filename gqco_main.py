import time
from datetime import datetime

from gqco.utils import fix_seed, print0, save_file
from gqco.initialize import initialize_all
from gqco.task import GQCO
from gqco.model import define_model
from gqco.train import train





if __name__ == '__main__':

    ## Initialize
    args = initialize_all()

    ## Set seed
    fix_seed(args.seed)

    ## Task definition
    gqeco = GQCO(args)
    args = gqeco.make_taskargs(args)

    ## Model definition
    model, args = define_model(args)

    ## Save settings
    save_file(obj={'task': gqeco, 'args': args}, file=f'{args.out_dir}/taskobjects.pkl', args=args)

    ## Training
    model = train(gqeco, args, model)

    ## END
    print0(f'Start time: {args.start_date}', args, terminal=True)
    print0(f'End time: {datetime.now().strftime("%Y/%m/%d/%H:%M:%S")}', args, terminal=True)
    print0(f'Running time [s]: {time.time() - args.start_time}', args, terminal=True)
    print0('FINISH :)', args, terminal=True)
