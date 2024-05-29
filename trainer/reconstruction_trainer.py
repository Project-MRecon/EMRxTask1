import numpy as np
from base.base_trainer import BaseTrainer
from utils.utils import show_3d_image, judge_log
from torch.optim.swa_utils import SWALR

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
        input, target, _, _ = self._load_and_visualize_input(batch_data, batch_idx, epoch)
        pred = self.model(input)
        loss = self.criterion(pred, target)
        self._update_metrics(pred, target)
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            step = epoch*self.len_epoch+(batch_idx // self.writer_step)
            self.writer.add_image("Train/Pred", show_img(pred.detach().cpu()), step, dataformats="HW")
        return loss
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch):
        input, target, mean, std = self._load_and_visualize_input(batch_data, batch_idx, epoch)
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
    # 未完待续
    def __init__(self, plans, data_loader, valid_data_loader=None):
        super().__init__(plans, data_loader, valid_data_loader)
        self.writer_step = 2

    def _load_and_visualization_input(self, batch_data, batch_idx, epoch, split):
        masked_kspace, mask, num_low_frequencies, kspace = (
            batch_data["masked_kspace"],
            batch_data["mask"],
            batch_data["num_low_frequencies"],
            batch_data["kspace"]
        )
        return super()._load_and_visualization_input()