import numpy as np
import torch
from torch.optim.swa_utils import SWALR
import fastmri

from base.base_trainer import BaseTrainer
from utils.utils import show_3d_image, judge_log


def show_img(batch_img):
    img = batch_img[0,0].numpy()
    return (img - np.min(img)) / (np.max(img) - np.min(img))


class ReconStructionTrainer(BaseTrainer):
    def __init__(self, plans, data_loader, valid_data_loader=None):
        super().__init__(plans, data_loader, valid_data_loader)
        self.writer_step = 2

    def _load_and_visualize_input(self, batch_data, batch_idx, epoch, split):
        input, target, mean, std = (
                batch_data["kspace_masked_ifft"],
                batch_data["reconstruction_rss"],
                batch_data["mean"],
                batch_data["std"],
            )
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            step = epoch*self.len_epoch+(batch_idx // self.writer_step)
            self.writer.add_image(f"{split}/Trans_Input", show_img(input), step, dataformats="HW")
            self.writer.add_image(f"{split}/Target", show_img(target), step, dataformats="HW")
            self.writer.add_image(f"{split}/Ori_Input", show_img(batch_data["kspace_masked_ifft_copy"]), step, dataformats="HW")
        
        input, target, mean, std = (
                input.to(self.device),
                target.to(self.device),
                batch_data["mean"].to(self.device),
                batch_data["std"].to(self.device),
            )
        return input, target, mean, std
        
    def _train_batch_step(self, batch_data, batch_idx, epoch):
        input, target, _, _ = self._load_and_visualize_input(batch_data, batch_idx, epoch, "Train")
        pred = self.model(input)
        loss = self.criterion(pred, target)
        self._update_metrics(pred, target)
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            step = epoch*self.len_epoch+(batch_idx // self.writer_step)
            self.writer.add_image("Train/Pred", show_img(pred.detach().cpu()), step, dataformats="HW")
        return loss
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch):
        input, target, mean, std = self._load_and_visualize_input(batch_data, batch_idx, epoch, "Valid")
        # 上级函数已经有@torch.no_grad()
        pred = self.model(input)
        # calulate loss
        loss = self.criterion(pred, target)
        self._update_metrics(pred, target)
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            step = epoch*self.len_epoch+(batch_idx // self.writer_step)
            self.writer.add_image("Valid/Pred", show_img(pred.detach().cpu()), step, dataformats="HW")
        return loss.item() # 返回item减少显存


class KspaceTrainer(BaseTrainer):
    '''
    还未测试
    如何把不同模态的不同大小的K空间变成一致大小放到一个batch里训练
    crop?pad?一个模态一个dataloader再合并?
    '''
    def __init__(self, plans, data_loader, valid_data_loader=None):
        super().__init__(plans, data_loader, valid_data_loader)
        self.writer_step = 2

    def Kspace2Image(self, kspace):
        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)

    def _load_and_visualize_input(self, batch_data, batch_idx, epoch, split):
        kspace, mask, num_low_frequencies, target= (
            batch_data["kspace"].to(self.device),
            batch_data["mask"].to(self.device).transpose(1,2).unsqueeze(1),
            batch_data["num_low_frequencies"].to(self.device),
            batch_data["reconstruction_rss"].to(self.device)
        )
        kspaces = [kspace[:,i,...] for i in range(5)]
        kspace = torch.concatenate(kspaces, dim=1)
        masked_kspace = kspace * mask
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), axis=-1)
        # undersample_img, target = self.Kspace2Image(masked_kspace), self.Kspace2Image(kspace)
        # if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
        #     step = epoch*self.len_epoch+(batch_idx // self.writer_step)
        #     self.writer.add_image(f"{split}/Undersample_img", show_img(undersample_img), step, dataformats="HW")
        #     self.writer.add_image(f"{split}/Target", show_img(target), step, dataformats="HW")


        return masked_kspace.float(), mask.float(), num_low_frequencies, target.float()
    
    def _train_batch_step(self, batch_data, batch_idx, epoch):
        masked_kspace, mask, num_low_frequencies, target = self._load_and_visualize_input(batch_data, batch_idx, epoch, "Train")
        pred = self.model(masked_kspace, mask, 16)
        loss = self.criterion(pred, target) # calulate loss
        self._update_metrics(pred, target) # calulate metrics
        return loss
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch):
        masked_kspace, mask, num_low_frequencies, target = self._load_and_visualize_input(batch_data, batch_idx, epoch, "Valid")
        # 上级函数已经有@torch.no_grad()
        pred = self.model(masked_kspace, mask, 16)
        # calulate loss
        loss = self.criterion(pred, target)
        self._update_metrics(pred, target)
        return loss