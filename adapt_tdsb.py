import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

from utils import neg_sisdr_loss_wrapper
from data.adapt_dataset import get_dataloaders
from models.speakerbeam import TimeDomainSpeakerBeam


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    return args


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SpeakerBeamLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TimeDomainSpeakerBeam(
            **config['filterbank'],
            **config['masknet'],
            sample_rate=config['dataset']['sample_rate'],
            **config['enroll']
        )
        self.config = config
        if self.config['adaptation']['oracle']:
            print('####### Using oracle adaptation #######')
        self.criterion = neg_sisdr_loss_wrapper
        self.save_hyperparameters(config)  # Saves config for checkpointing

    def forward(self, mixture, enrollment):
        return self.model(mixture, enrollment)

    def training_step(self, batch, batch_idx):
        mixture = batch['noisy']
        target = batch['clean'] if self.config['adaptation']['oracle'] else batch[self.config['adaptation']['teacher']]
        enrollment = batch['enrollment']
        output = self(mixture, enrollment)
        loss = self.criterion(output, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=mixture.size(0))

        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            self.log('learning_rate', current_lr)
        return loss

    def validation_step(self, batch, batch_idx):
        mixture = batch['noisy']
        target = batch['clean']
        enrollment = batch['enrollment']
        output = self(mixture, enrollment)
        loss = self.criterion(output, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=mixture.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = make_optimizer(self.model.parameters(), **self.config['optim'])
        return optimizer
        # if self.config['train']['half_lr']:
        #     scheduler = {
        #         'scheduler': ReduceLROnPlateau(
        #             optimizer=optimizer,
        #             factor=0.5,
        #             patience=self.config['scheduler']['lr_reduce_patience']
        #         ),
        #         'monitor': 'val_loss',
        #         'interval': 'epoch',
        #         'frequency': 1,
        #         'strict': True,
        #     }
        #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # else:
        #     return optimizer


class SpeakerBeamDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader, _ = get_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(config['seed'])
    torch.set_float32_matmul_precision('medium') # set to 'medium' or 'high'

    # Create necessary directories
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)

    # Instantiate the LightningModule
    if 'adapt' not in config['checkpoint']['resume']:
        model = SpeakerBeamLightningModule.load_from_checkpoint(
            checkpoint_path=config['checkpoint']['resume'],
            config=config,
            strict=True,
        )
    
    else:
        model = SpeakerBeamLightningModule(config)

    # Instantiate the DataModule
    data_module = SpeakerBeamDataModule(config)

    # Callbacks
    callbacks = []

    # EarlyStopping
    early_stopping_callback = EarlyStopping(
        monitor=config['early_stopping']['monitor'],
        patience=config['early_stopping']['patience'],
        verbose=config['early_stopping']['verbose'],
        mode=config['early_stopping']['mode'],
        min_delta=config['early_stopping']['delta'],
    )
    callbacks.append(early_stopping_callback)

    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint']['dir'],
        filename=config['checkpoint']['ckpt_name'],
        save_top_k=config['checkpoint']['save_best'],
        save_last=config['checkpoint']['save_last'],
        verbose=config['checkpoint']['verbose'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
    )
    callbacks.append(checkpoint_callback)

    # LearningRateMonitor
    if config['train']['log_lr']:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config['train']['log_dir'],
        name='lightning_logs',
        version='0',
    )

    # Determine if DDP should be used
    ddp_config = config.get('ddp', {})
    use_ddp = ddp_config.get('use_ddp', False)
    num_nodes = ddp_config.get('num_nodes', 1)
    num_gpus = ddp_config.get('num_gpus', torch.cuda.device_count())
    strategy = ddp_config.get('strategy', 'ddp')
    strategy = DDPStrategy(find_unused_parameters=True)

    if use_ddp and num_gpus > 1:
        # Explicitly set DDP strategy and related arguments
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu',
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
        )
    else:
        # Single GPU or CPU
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
        )

    # Resume from checkpoint if specified
    if 'adapt' in config['checkpoint']['resume']:
        print(f"####### Resuming from checkpoint {config['checkpoint']['resume']} #######")
        ckpt_path = config['checkpoint']['resume']
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        # Fit the model
        trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()