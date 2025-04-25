
# # dataset.py
# import torch
# from torch.utils.data import Dataset
# import cv2
# import os
# import numpy as np
#
# class InpaintDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, gt_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.gt_paths = gt_paths
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image = cv2.imread(self.image_paths[idx])
#         mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
#         gt = cv2.imread(self.gt_paths[idx])
#
#         # Convert to RGB and normalize
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
#         mask = mask / 255.0
#         gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255.0
#
#         # Convert to torch tensors
#         image = torch.from_numpy(image).permute(2, 0, 1).float()
#         mask = torch.from_numpy(mask).unsqueeze(0).float()  # add channel dimension
#         gt = torch.from_numpy(gt).permute(2, 0, 1).float()
#
#         # Concatenate image and mask to create 4-channel input
#         input_tensor = torch.cat([image, mask], dim=0)
#
#         return input_tensor, gt, mask


# dataset.py
# image, mask, gt가 서로 다른 폴더에 있어도 되나 같은 것에 해당하는 파일은 이름이 아예 같도록 저장돼있어야 하는 코드

import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class InpaintDataset(Dataset):
    def __init__(self, image_dir, mask_dir, gt_dir,large_mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.large_mask_dir = large_mask_dir
        self.transform = transform

        # Load all file names from the image directory (assuming same filenames exist in mask and gt directories)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)
        large_mask_path = os.path.join(self.large_mask_dir, filename)

        # Load the images
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE 옵션을 빼면 BGR 형식으로 읽어 오류가 날 수 있음
        gt = cv2.imread(gt_path)
        large_mask = cv2.imread(large_mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")
        if gt is None:
            raise FileNotFoundError(f"Could not read ground truth at {gt_path}")


        # Convert to RGB and normalize
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # [0, 1] 범위로 정규화
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 127.5) - 1.0  # [-1, 1] 범위로 정규화
        mask = mask / 255.0
        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255.0    # [0, 1] 범위로 정규화
        gt = (cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 127.5) - 1.0  # [-1, 1] 범위로 정규화
        large_mask = large_mask / 255.0

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # torch.from_numpy로 numpy -> tensor 변환
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # add channel dimension    # torch.from_numpy로 numpy -> tensor 변환
        # opencv로 읽어온 mask는 원래 2D numpy 배열이므로 2차원임. 채널 차원을 추가해줘야하니 unsqueeze 실행
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        large_mask = torch.from_numpy(large_mask).unsqueeze(0).float()

        # Concatenate image and mask to create 4-channel input
        # input_tensor = torch.cat([image, mask], dim=0)

        return image, gt, mask,filename, large_mask
