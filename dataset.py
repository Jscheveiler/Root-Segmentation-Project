import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RootSegmentationDataset(Dataset):
    """
    Dataset for root segmentation with RGB images and binary masks.
    Expects:
    - raw RGB images: data/1_mesocosm/raw/*.jpg
    - manual masks:   data/1_mesocosm/manual_masks/*.png
    Assumes matching filenames between raw and masks.

    This project being more of a Proof-Of-Concept, no data augmentation will be used (for faster training)
    """
    def __init__(self, root_dir="data/1_mesocosm"):
        super().__init__()
        self.image_dir = os.path.join(root_dir, "raw")
        self.mask_dir = os.path.join(root_dir, "manual_masks")
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale (255: bg, 0: root)

        # Convert to tensor and normalize mask
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        mask = (mask < 0.5).float()  # binary [0, 1]

        return image, mask
