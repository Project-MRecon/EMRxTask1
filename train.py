import os
import argparse

import torch
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel  # 随机权重平均
from monai.data import DataLoader

from trainer.reconstruction_trainer import ReconStructionTrainer
from utils.ddp_utils import init_distributed_mode
from utils.utils import init_seeds
from dataloading.generate_dataset import generate_train_val_ds
join = os.path.join


# 一些设置
SEED = 2023
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置显卡间通信方式
torch.backends.cudnn.deterministic = False  # 在gpu训练时固定随机源
torch.backends.cudnn.benchmark = True   # 搜索卷积方式，启动算法前期较慢，后期会快
# torch.autograd.detect_anomaly() # debug的时候启动，用于检查异常


def main(plans_args):
    '''
    设计思路抄自pytorch lighting 和 nnunetv2
    callback 只用在了early stopping, 目前没有其他使用需求
    装饰器只用了静态和抽象装饰器和timer
    没有实现梯度累加功能,如果实现记得在DDP模式下取消前几次的梯度同步
    '''

    plans_args = init_distributed_mode(plans_args)
    # 随机数设置一下
    if not plans_args.ddp:
        init_seeds(seed=SEED)
    else:
        init_seeds(seed=SEED+plans_args.rank)

    train_ds, val_ds, plans_args = generate_train_val_ds(plans_args)

    # dataloader总得设置一下吧
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) \
        if plans_args.ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds) \
        if plans_args.ddp else None
    train_dataloader = DataLoader(
        train_ds, shuffle=(train_sampler is None),
        batch_size=plans_args.batch_size, num_workers=8,
        sampler=train_sampler, pin_memory=True)

    val_dataloader = DataLoader(
        val_ds, shuffle=(val_sampler is None),
        batch_size=plans_args.batch_size, num_workers=8,
        sampler=val_sampler, pin_memory=True)

    trainer = ReconStructionTrainer(
        plans=plans_args,
        data_loader=train_dataloader,
        valid_data_loader=val_dataloader,
        )

    trainer.train()  # 也可以在这里传入model和dataloader
    if plans_args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    config = {
        "json_path": "/homes/syli/dataset/MultiCoil/test/test.json",
        "resume": None,  # "./saved/model/0807_1536/checkpoint-epoch48.pth"
        "epochs": 100,
        "save_dir": "./saved",
        'network': {"arch": "BasicUNet",
                    "args": {"spatial_dims": 2,
                            "in_channels": 1,
                            "out_channels": 1,
        "features": [32, 64, 128, 256, 512, 32],}},
        "optimizer": {
            "type": "Adam",
            "args": {
                "lr": 1e-2,
                "weight_decay": 1e-4,
                }},
        "batch_size": 4,
        "valid_interval": 2,
        "standard": "SSIM",
        "Metrics": {
            "SSIM": {"spatial_dims": 2,
                            "data_range": 6}},
        "early_stopping": 20,
        "criterion": "L1Loss",
        "use_amp": True
    }

    plans_args = argparse.Namespace(**config)
    plans_args.config = config
    main(plans_args)
