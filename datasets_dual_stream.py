from glob import glob
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import torch
from torchvision import transforms as T

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        
        # Extract frame index (assuming format frame000123.jpg)
        frame_number = int(os.path.basename(frame_path).split('.')[0].replace('frame_', ''))
        
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        frame = Image.open(frame_path).convert("RGB")
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label, video_name, frame_number, frame_path


class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.flow_paths = sorted(glob(f'{root_dir}/flows_png/{split}/*/*/*.png'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform

        # Build a dictionary for fast lookup
        self.flow_dict = {}  # (video_name, i, j) -> flow_path
        for path in self.flow_paths:
            video_name = path.split('/')[-2]
            basename = os.path.basename(path).split('.')[0]
            i, j = map(int, re.findall(r'\d+', basename))
            self.flow_dict[(video_name, i, j)] = path

    def __len__(self):
        return len(self.flow_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        flow_path = self.flow_paths[idx]
        video_name = flow_path.split('/')[-2]
        basename = os.path.basename(flow_path).split('.')[0]
        i, j = map(int, re.findall(r'\d+', basename))

        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        flow_img = Image.open(flow_path).convert("RGB")
        if self.transform:
            flow_img = self.transform(flow_img)
        else:
            flow_img = T.ToTensor()(flow_img)

        return flow_img, label, video_name, (i, j), flow_path

    def get_flow(self, video_name, i, j):
        """Fast lookup of flow image path by video_name and frame indices."""
        return self.flow_dict.get((video_name, i, j), None)


class DualStreamDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dataset, flow_dataset, window_size=2):
        self.frame_dataset = frame_dataset
        self.flow_dataset = flow_dataset
        self.window_size = window_size

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frame, label, video_name, frame_number, _ = self.frame_dataset[idx]

        # Build flow pairs around the frame_number
        flow_imgs = []
        for i in range(frame_number - self.window_size + 1, frame_number + self.window_size):
            j = i + 1
            if i >= 1:  # Ensure valid indices (no zero or negative)
                flow_path = self.flow_dataset.get_flow(video_name, i, j)
                if flow_path:
                    flow_img = Image.open(flow_path).convert("RGB")
                    if self.frame_dataset.transform:
                        flow_img = self.frame_dataset.transform(flow_img)
                    else:
                        flow_img = T.ToTensor()(flow_img)
                    flow_imgs.append(flow_img)

        # Ensure consistent tensor shape even if some flows are missing
        num_flows = self.window_size * 2
        if len(flow_imgs) == 0:
            # No flows available
            flow_stack = torch.zeros((num_flows, 3, frame.shape[1], frame.shape[2]))
        elif len(flow_imgs) < num_flows:
            # Pad missing flows with zeros
            padding = torch.zeros((num_flows - len(flow_imgs), 3, frame.shape[1], frame.shape[2]))
            flow_stack = torch.cat([torch.stack(flow_imgs), padding], dim=0)
        else:
            flow_stack = torch.stack(flow_imgs)

        return frame, flow_stack, label
