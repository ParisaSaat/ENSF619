import os

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BrainMRI2D(Dataset):
    def __init__(self, img_root_dir, gt_root_dir=None, file_ids=None, transform=None, labeled=True):
        self.img_root_dir = img_root_dir
        self.gt_root_dir = gt_root_dir
        self.file_ids = file_ids
        self.transform = transform
        self.labeled = labeled

        self.pairs_path = []
        for file_id in self.file_ids:
            img_path = os.path.join(self.img_root_dir, file_id)
            gt_path = os.path.join(self.gt_root_dir, file_id) if self.labeled else None
            self.pairs_path.append((img_path, gt_path))

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs_path[idx]
        nifti_image = nib.load(img_path)
        image_affine = nifti_image.affine
        image = nifti_image.get_fdata(dtype=np.float32)
        nifti_mask = nib.load(mask_path)
        mask = nifti_mask.get_fdata(dtype=np.float32)
        mask_affine = nifti_mask.affine

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed.get('image')
            mask = transformed.get('mask')
        sample = {'image': image, 'mask': mask, 'mask_affine': mask_affine, 'image_affine': image_affine, 'idx': idx}
        return sample

    def __len__(self):
        return len(self.pairs_path)
