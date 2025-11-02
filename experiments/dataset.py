from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UCF101Dataset(Dataset):
    def __init__(
        self, data_dir, split_file, max_frames=16, frame_size=112, class_to_idx=None
    ):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.samples = []

        with open(split_file) as f:
            lines = f.readlines()

        if class_to_idx is None:
            self._setup_training_mode(lines)
        else:
            self._setup_validation_mode(lines, class_to_idx)

    def _setup_training_mode(self, lines):
        self._build_class_mapping(lines)
        self._load_labeled_samples(lines)

    def _setup_validation_mode(self, lines, class_to_idx):
        self.class_to_idx = class_to_idx
        self._load_mixed_format_samples(lines)

    def _build_class_mapping(self, lines):
        self.class_to_idx = {}
        class_names_seen = set()

        for line in lines:
            if " " in line:
                path = line.strip().split(" ")[0]
                class_name = path.split("/")[0]
                if class_name not in class_names_seen:
                    class_names_seen.add(class_name)
                    self.class_to_idx[class_name] = len(self.class_to_idx)

    def _load_labeled_samples(self, lines):
        for line in lines:
            if " " in line:
                path = line.strip().split(" ")[0]
                class_name = path.split("/")[0]
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    self.samples.append((path, label))

    def _load_mixed_format_samples(self, lines):
        for line in lines:
            if " " in line:
                path = line.strip().split(" ")[0]
            else:
                path = line.strip()

            if path:
                class_name = path.split("/")[0]
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
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
                frames.append(
                    frames[-1]
                    if frames
                    else np.zeros((self.frame_size, self.frame_size, 3))
                )
        else:
            frames = frames[: self.max_frames]

        # Convert to tensor (T, H, W, C) -> (T, C*H*W)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).reshape(self.max_frames, -1)

        return frames, label
