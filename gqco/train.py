
import wandb
import psutil
import os

import cudaq
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from gqco.utils import print0, fix_seed
from gqco.evaluation import make_log, check_accuracy
from gqco.loss import compute_loss
from gqco.model import grad_false_for_unused_experts, param_clone, grad_true_for_expert_tuning
from gqco.data import DummyDataset, generate_data






class MyModel(pl.LightningModule):

    def __init__(self, model, task, args):
        super(MyModel, self).__init__()
        self.model = model
        self.task = task
        self.args = args

        self.current_size = args.start_size
        self.save_tag = 0

        if args.quantum_tool == 'cudaq':
            print(f'{cudaq.get_target().name}')


    def forward(self, record, temperature=1):
        return self.model.forward(
            record,
            masked_tokens=self.task.bad_tokens[record['size']], 
            temperature=temperature,
            same_token_penalty=0.0
        )
    

    def training_step(self, batch, batch_idx):

        ## Setups
        epoch = int(self.current_epoch + self.args.start_epoch)
        self.args.epoch = epoch
        self.args.device = self.device
        self.args.rank = self.trainer.strategy.global_rank

        ## Random seeds
        seed_for_generation = int(((epoch+1)*373 + self.trainer.strategy.global_rank) % 2**30)
        seed_for_training = int(((epoch+1)*373 + self.trainer.strategy.global_rank) % 2**30)

        # print0(f'---------- Epoch {epoch}----------', self.args)


        ## Start training
        self.model.train()


        ### Data generation
        adj, size, record = generate_data(self.args, 
                                          num_clone=self.args.num_policy_search, 
                                          seed=seed_for_generation, 
                                          current_size=self.current_size,
                                          device=self.device)

        
        if self.args.tune_size < 0:
            ### Set requires_grad=False for unused expart layers
            self.model = grad_false_for_unused_experts(self.model, size, self.args)      

        if self.args.tune_size > 0:
            ### Set requires_grad=True for target expart layers
            self.model = grad_true_for_expert_tuning(self.model, self.args.tune_size, self.args)


        ### Token generation
        fix_seed(seed_for_training)
        out_tokens, probs_all, _, _ = self(record)

        ### Loss computation
        loss, log_psum_best, min_energy, max_energy, mean_energy = compute_loss(self.task, adj, out_tokens, probs_all, self.args)


        ## Logging
        self.log('training loss', loss, sync_dist=True)
        self.log('min energy', min_energy, sync_dist=True)
        self.log('mean energy', mean_energy, sync_dist=True)
        self.log('max energy', max_energy, sync_dist=True)
        self.log('log probability of the best', log_psum_best, sync_dist=True)


        return loss






    def validation_step(self, batch, batch_idx):

        self.model.eval()

        self.args.device = self.device
        self.args.rank = self.trainer.strategy.global_rank

        epoch = int(self.current_epoch + self.args.start_epoch)
        seed_for_eval = int(((epoch+42)*373 + self.trainer.strategy.global_rank) % 2**30)


        log_dict = {'current_size': self.current_size}

        
        ## Performance evaluation
        is_update = False
        if self.current_epoch % self.args.log_freq == self.args.log_freq - 1:
            
            print0(f'---------- Epoch {int(self.current_epoch + self.args.start_epoch)}----------', self.args)

            log_dict = make_log(self.args, self.model, self.task, log_dict=log_dict, seed=seed_for_eval, current_size=self.current_size)


        ## Check accuracy
        if self.current_epoch % self.args.log_freq == self.args.log_freq - 1:
            log_dict, accs = check_accuracy(self.args, self.model, self.task, log_dict=log_dict, seed=seed_for_eval, num_problems=10)
            """
                acc th
            """
            if self.args.tune_size < 0:
                min_acc = 1.1
                for s in range(3, log_dict['current_size']+1):
                    acc = log_dict[f'Accuracy (size: {s})']
                    acc_gather = self.all_gather(acc).mean()
                    if acc_gather < min_acc:
                        min_acc = acc_gather


                if min_acc >= 0.9:
                    is_update = True
                log_dict[f'Accuracy (monitor)'] = min_acc

        
        for k, v in log_dict.items():
            self.log(k, v, sync_dist=True)
        
        ## Update size
        if is_update:

            self.model = param_clone(self.model, self.current_size, self.current_size+1, self.device)

            self.current_size += 1
            self.current_size = min(self.current_size, self.args.max_size)




    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.args.learning_rate, 
            betas = (0.9, 0.95), 
            eps = 1e-8, 
            weight_decay = 0
        )

        return optimizer
    

    def train_dataloader(self):
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=1, shuffle=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_size'] = self.current_size

    def on_load_checkpoint(self, checkpoint):
        self.current_size = checkpoint['current_size']
    






def train(task, args, model, data_generation_func=None):

    for p in model.parameters():
        p.requires_grad = True

    ## WandB
    if args.is_wandb:
        wandb_logger = WandbLogger(
            project = args.project_name, 
            name = args.task_name+'-'+str(args.job_id)+'-'+args.start_date,
            config = args
        )
    else:
        wandb_logger = WandbLogger(mode="disabled")


    ## Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.checkpoint_dir,
        save_top_k = -1,
        verbose = True,
        monitor = None,
        every_n_train_steps=args.checkpoint_freq,
        filename = 'latest-{current_size:.0f}-{epoch}'
    )

    ## Load checkpoint
    if (args.init_checkpoint is not None) & (args.init_checkpoint != 'None'):
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        args.start_epoch = checkpoint['epoch'] + 1
        pl_model = MyModel.load_from_checkpoint(checkpoint_path=args.init_checkpoint, model=model, task=task, args=args, map_location="cpu")
        torch.cuda.empty_cache()
    else:
        pl_model = MyModel(model=model, task=task, args=args)
        args.start_epoch = 0


    ## Define trainer
    trainer = Trainer(
        max_epochs = args.max_epoch,
        logger = wandb_logger,
        callbacks = [checkpoint_callback],
        log_every_n_steps = 1,
        enable_progress_bar=False,
        accelerator='auto',
        devices = -1,
        num_nodes = args.num_hosts,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true'
    )
    torch.cuda.empty_cache()

    ## Training
    trainer.fit(pl_model)


    wandb.finish()

    return pl_model
