import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path


class UCF101Dataset(Dataset):
    def __init__(self, data_dir, split_file, max_frames=16, frame_size=112, num_classes=5):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.frame_size = frame_size

        # Load split file
        with open(split_file) as f:
            lines = f.readlines()

        # Only use first few classes for quick training
        self.samples = []
        self.class_to_idx = {}

        for line in lines:
            if ' ' in line:
                path, label = line.strip().split(' ')
                label = int(label) - 1  # 0-indexed
                if label < num_classes:
                    self.samples.append((path, label))
                    class_name = path.split('/')[0]
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = label
            elif line.strip():  # Test split format (no label)
                path = line.strip()
                class_name = path.split('/')[0]
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    if label < num_classes:
                        self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video_path = self.data_dir / "UCF-101" / path

        # Load video frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and normalize
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            # Repeat last frame
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3)))
        else:
            frames = frames[:self.max_frames]

        # Convert to tensor (T, H, W, C) -> (T, C*H*W)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).reshape(self.max_frames, -1)

        return frames, label