import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
import multiprocessing
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
import datetime

from trainer import Trainer
from config import Config
from utils import (
    get_model,
    set_grad,
    get_preprocess,
    get_laion_cirr_dataset,
    # get_laion_only_dataset,    # Uncomment if you implement
    collate_fn,
    extract_index_features,
    get_optimizer
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def main(cfg):
    # setup_seed(cfg.seed)   # optionally use seed from cfg
    
    # Load model & set gradient config
    model = get_model(cfg)
    set_grad(cfg, model)
    model.pretrained_model.eval().float()
    
    if cfg.model_name.lower().startswith('blip'):
        input_dim = 384
    elif cfg.model_name.lower().startswith('clip'):
        input_dim = model.pretrained_model.visual.input_resolution
    else:
        raise ValueError(f"Unsupported model_name {cfg.model_name}")
    
    preprocess = get_preprocess(cfg, model, input_dim)
    
    # Dataset loading: TRAIN ONLY
    if cfg.skip_eval:
        # Only training on Laion-CIR triplets
        # If you have get_laion_only_dataset implemented:
        # relative_train_dataset = get_laion_only_dataset(preprocess, cfg.laion_type)
        # Otherwise use get_laion_cirr_dataset but take only first return
        relative_train_dataset, _, _ = get_laion_cirr_dataset(preprocess, cfg.laion_type,  skip_eval=True)
        relative_val_dataset = None
        classic_val_dataset = None
    else:
        # Evaluation branch (you don’t use now)
        relative_train_dataset, relative_val_dataset, classic_val_dataset = get_laion_cirr_dataset(preprocess, cfg.laion_type, skip_eval=False)
    
    # Build DataLoader for training
    assert relative_train_dataset is not None, "Training dataset not loaded!"
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=cfg.batch_size,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True
    )
    
    # Prepare optional components for evaluation (will be None in train-only mode)
    kwargs = {}
    if (not cfg.skip_eval) and (cfg.dataset.lower() == 'cirr') and (cfg.encoder in ['text','neither']):
        # Only compute index features if eval is going to happen
        val_index_features, val_index_names, val_total_index_features = extract_index_features(
            classic_val_dataset, model, return_local=False
        )
        kwargs['val_index_features'] = val_index_features
        kwargs['val_index_names'] = val_index_names
        kwargs['val_total_index_features'] = val_total_index_features
    else:
        kwargs['val_index_features'] = None
        kwargs['val_index_names'] = None
        kwargs['val_total_index_features'] = None
    
    # Setup optimizer, scheduler, loss
    optimizer = get_optimizer(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.num_epochs,
        eta_min=1e-2 * cfg.learning_rate,
        last_epoch=-1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Create Trainer and start training
    trainer = Trainer(
        cfg, model, relative_train_loader,
        optimizer, lr_scheduler, criterion,
        classic_val_dataset, relative_val_dataset,
        **kwargs
    )
    trainer.train()

if __name__ == '__main__':
    cfg = Config()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.save_path = f"{cfg.save_path_prefix}/{current_time}_{cfg.comment}_best_arithmetic.pth"
    
    wandb_config = vars(cfg)
    wandb.init(
        project='ZeroShot-CIR',
        notes=cfg.comment,
        config=wandb_config,
        name=cfg.comment
    )
    
    main(cfg)
    wandb.finish()
