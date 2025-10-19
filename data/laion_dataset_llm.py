from torch.utils.data import Dataset
import json
import PIL
from PIL import Image, ImageFile
import os

data_file_path = os.path.dirname(__file__)
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _resolve_with_ext(root, stem):
    """
    Given an ID '0001234' (no ext), return an existing path by trying
    .png first (your case), then .jpg/.jpeg.
    """
    cand = os.path.join(root, f"{stem}.png")
    if os.path.exists(cand):
        return cand
    for ext in (".jpg", ".jpeg"):
        cand = os.path.join(root, f"{stem}{ext}")
        if os.path.exists(cand):
            return cand
    return None  # not found

class LaionDataset_LLM(Dataset):
    def __init__(self, split: str, preprocess: callable, image_root: str):
        self.preprocess = preprocess
        self.split = split
        if split not in ['train']:
            raise ValueError("split should be in ['train']")
        self.image_root = image_root

        with open(os.path.join(data_file_path, "files", "laion_llm_info.json")) as f:
            image_ids_map = json.load(f)

        # Build a vetted list of valid samples (paths exist)
        self.samples = []
        missing = 0

        # Keys are reference IDs (strings); values contain tgt_image_id & caption
        for ref_key, val in image_ids_map.items():
            ref_id = str(ref_key).zfill(7)
            tgt_id = str(val['tgt_image_id']).zfill(7)

            ref_path = _resolve_with_ext(self.image_root, ref_id)
            tgt_path = _resolve_with_ext(self.image_root, tgt_id)

            if (ref_path is None) or (tgt_path is None):
                missing += 1
                continue

            caption = val.get('relative_cap', val.get('edit_cap', ''))
            self.samples.append((ref_path, tgt_path, caption))

        kept = len(self.samples)
        print(f"Laion {split} dataset initialized (root={self.image_root})")
        print(f"Kept {kept} pairs; skipped {missing} missing pairs.")

        if kept == 0:
            raise RuntimeError(
                "No valid (ref, tgt) pairs found. Check laion_image_root, "
                "file extensions, and that IDs in laion_llm_info.json match filenames."
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
