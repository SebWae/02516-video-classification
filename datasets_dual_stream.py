from glob import glob
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.transform = transform

        # Group frames by video
        self.video_to_frames = {}
        for frame_path in sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg')):
            video_name = frame_path.split('/')[-2]
            self.video_to_frames.setdefault(video_name, []).append(frame_path)

        # Unique videos only
        self.videos = list(self.video_to_frames.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Select video by index
        video_name = self.videos[idx]

        # Choose ONE random frame from this video
        frame_paths = self.video_to_frames[video_name]
        frame_path = np.random.choice(frame_paths)

        label = self.df.loc[self.df['video_name'] == video_name, 'label'].item()

        frame = Image.open(frame_path).convert('RGB')
        frame = self.transform(frame) if self.transform else T.ToTensor()(frame)

        return frame, label, video_name
    

class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.transform = transform

        # grouping flows by video
        self.video_to_flows = {}
        for path in sorted(glob(f'{root_dir}/flows_png/{split}/*/*/*.png')):
            video_name = path.split('/')[-2]
            self.video_to_flows.setdefault(video_name, []).append(path)

        self.videos = list(self.video_to_flows.keys())

    def get_entire_flow_stack(self, video_name):
        flow_paths = self.video_to_flows[video_name]
        flow_imgs = []
        for flow_path in flow_paths:
            flow_img = Image.open(flow_path).convert('RGB')
            flow_img = self.transform(flow_img) if self.transform else T.ToTensor()(flow_img)
            flow_imgs.append(flow_img)
        return torch.stack(flow_imgs) if flow_imgs else None


class DualStreamDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dataset, flow_dataset):
        self.frame_dataset = frame_dataset
        self.flow_dataset = flow_dataset
        self.videos = frame_dataset.videos  # both datasets must share same video list

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Get a random frame from this video
        frame, label, video_name = self.frame_dataset[idx]

        # Get the entire flow stack for this video
        flow_stack = self.flow_dataset.get_entire_flow_stack(video_name)

        # Optionally ensure tensor format
        if flow_stack is None:
            flow_stack = torch.zeros((0, 3, frame.shape[1], frame.shape[2]))

        return frame, flow_stack, label
