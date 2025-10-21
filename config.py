import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    # ---- paths & dataset selection ----
    base_path: str = "/content/drive/MyDrive/Zero-shot"
    laion_type: str = "laion_combined"  # 'laion_llm' | 'laion_template' | 'laion_combined'
    laion_image_root: str = field(init=False)

    # Train-only flow
    skip_eval: bool = True
    dataset: str = "laion"         # we’re training on Laion only

    # ---- model / training ----
    dropout: float = 0.5
    num_layers: int = 2
    model_name: str = "clip-Vit-B/32"  # trained path in your code handles this
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = 8
    encoder: str = 'text'           # 'neither' | 'text' | 'both'
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 10
    save_best: bool = True
    use_amp: bool = True
    num_workers: int = 2            # safer on Colab
    validation_frequency: int = 1
    comment: str = "colab_run_clipB32_combined"

    # ---- saving ----
    save_path_prefix: str = "/content/drive/MyDrive/Zero-shot/model/checkpoints"
    eval_load_path: str = "/content/drive/MyDrive/Zero-shot/model/checkpoints/best.pth"
    submission_name: str = 'colab_cirr_test_run'
    checkpoint_dir: str = "/content/drive/MyDrive/Zero-shot/model/"

    def __post_init__(self):
        # images are inside /Zero-shot/laion_chatgpt_16k  (change to ".../images" if that's your layout)
        self.laion_image_root = f"{self.base_path}/laion_cir_combined"
