from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from utils import set_train_bar_description, update_train_running_results


class Trainer:
    def __init__(self, cfg, model, train_dataloader, optimizer, scheduler, criterion,
                 classic_val_dataset=None, relative_val_dataset=None, **kwargs):
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = cfg.device
        self.use_amp = cfg.use_amp
        self.num_epochs = cfg.num_epochs
        self.checkpoint_dir = getattr(cfg, "checkpoint_dir", cfg.save_path_prefix)
        self.save_best = getattr(cfg, "save_best", True)

        # optional datasets (unused here)
        self.classic_val_dataset = classic_val_dataset
        self.relative_val_dataset = relative_val_dataset

        # AMP
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        print(f"[Trainer] Initialized for {self.num_epochs} epochs. AMP={self.use_amp}")

    def train(self):
        """Train-only version; no evaluation."""
        self.model.to(self.device)
        best_score = float('-inf')

        for epoch in range(self.num_epochs):
            epoch_loss = self.train_epoch(epoch)

            # Save checkpoint every epoch
            torch.save(self.model.state_dict(),
                       f"{self.checkpoint_dir}/epoch_{epoch+1:02d}.pth")

            # Log to wandb
            wandb.log({"epoch": epoch, "train_loss": epoch_loss})

            print(f"[Epoch {epoch+1}/{self.num_epochs}] Avg Train Loss: {epoch_loss:.4f}")

        print("Training complete. All checkpoints saved.")

    def train_epoch(self, epoch):
        self.model.train()
        train_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(self.train_dataloader, ncols=140)
        iters = len(train_bar)

        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(self.device, non_blocking=True)
            target_images = target_images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if not self.use_amp:
                logits = self.model(captions, reference_images, target_images)
                ground_truth = torch.arange(images_in_batch, device=self.device)
                loss = self.criterion(logits, ground_truth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                self.scheduler.step(epoch + idx / iters)
            else:
                with torch.amp.autocast('cuda'):
                    logits = self.model(captions, reference_images, target_images)
                    ground_truth = torch.arange(images_in_batch, device=self.device)
                    loss = self.criterion(logits, ground_truth)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.scaler.step(self.optimizer)
                self.scheduler.step(epoch + idx / iters)
                self.scaler.update()

            update_train_running_results(train_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, self.num_epochs, train_results)

        avg_loss = float(train_results['accumulated_train_loss'] / train_results['images_in_epoch'])
        return avg_loss
