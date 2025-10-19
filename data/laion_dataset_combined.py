# data/laion_dataset_combined.py
from torch.utils.data import Dataset
import json
import PIL
from PIL import Image, ImageFile
import os

data_file_path = os.path.dirname(__file__)
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _resolve_with_ext(root, stem):
    """Try .png first, then .jpg/.jpeg."""
    cand = os.path.join(root, f"{stem}.png")
    if os.path.exists(cand):
        return cand
    for ext in (".jpg", ".jpeg"):
        p = os.path.join(root, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None

class LaionDataset_Combined(Dataset):
    def __init__(self, split: str, preprocess: callable, image_root: str):
        self.preprocess = preprocess
        self.split = split
        if split not in ['train']:
            raise ValueError("split should be in ['train']")
        self.image_root = image_root  # <- IMPORTANT: accept external root

        # Combined mapping built by you:
        json_path = os.path.join(data_file_path, "files", "laion_combined_info.json")
        with open(json_path) as f:
            image_ids_map = json.load(f)

        self.samples = []
        missing = 0

        for ref_key, val in image_ids_map.items():
            # Handle both numeric and already-padded IDs
            ref_id_raw = str(ref_key)
            tgt_id_raw = str(val['tgt_image_id'])
            ref_id = ref_id_raw.zfill(7) if len(ref_id_raw) < 7 else ref_id_raw
            tgt_id = tgt_id_raw.zfill(7) if len(tgt_id_raw) < 7 else tgt_id_raw

            ref_path = _resolve_with_ext(self.image_root, ref_id)
            tgt_path = _resolve_with_ext(self.image_root, tgt_id)
            if (ref_path is None) or (tgt_path is None):
                missing += 1
                continue

            caption = val.get('relative_cap', val.get('edit_cap', ''))
            self.samples.append((ref_path, tgt_path, caption))

        kept = len(self.samples)
        print(f"Laion {split} dataset (combined) initialized (root={self.image_root})")
        print(f"Kept {kept} pairs; skipped {missing} missing pairs.")
        if kept == 0:
            raise RuntimeError(
                "No valid pairs found for combined. Check laion_image_root and laion_combined_info.json."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ref_path, tgt_path, caption = self.samples[index]
        reference_image = PIL.Image.open(ref_path).convert("RGB")
        target_image    = PIL.Image.open(tgt_path).convert("RGB")
        reference_image = self.preprocess(reference_image)
        target_image    = self.preprocess(target_image)
        return reference_image, target_image, caption
