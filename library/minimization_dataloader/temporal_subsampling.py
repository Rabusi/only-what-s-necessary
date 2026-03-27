import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from Config import cfg



#######---------------------UCSD Ped dataset-------------------#######

class UCSD_Ped_dataset(Dataset):
    def __init__(self, root_dir, data_split, shuffle=True, data_percentage=1.0, sequence_size=10, stride=5, step=5, mode="full"):
        
        self.root_dir = root_dir
        self.data_split = data_split
        self.sequence_size = sequence_size
        self.stride = stride
        self.step = step
        self.mode = mode

        if mode == "full":
            self.target_size = (256, 256)   # (W, H)
        elif mode == "downsamplex2":
            self.target_size = (128, 128)
        elif mode == "downsamplex4":
            self.target_size = (64, 64)
        else:
            raise ValueError("mode must be 'full', 'downsamplex2', or 'downsamplex4'")

        split_path = os.path.join(self.root_dir, self.data_split)
        self.data = []   # list of (frame_paths, start_idx)

        valid_exts = (".tif", ".tiff", ".jpg", ".png")

        for dirpath, _, filenames in os.walk(split_path):
            frames = [fn for fn in filenames if fn.lower().endswith(valid_exts)]
            if not frames:
                continue

            frames.sort()
            frame_paths = [os.path.join(dirpath, fn) for fn in frames]
            n = len(frame_paths)

            clip_span = (self.sequence_size - 1) * self.stride + 1
            if n < clip_span:
                continue

            max_start = n - clip_span + 1
            for start in range(0, max_start, self.step):
                self.data.append((frame_paths, start))

        if shuffle:
            random.shuffle(self.data)

        limit = int(len(self.data) * data_percentage)
        self.data = self.data[:max(1, limit)]

        if len(self.data) == 0:
            raise RuntimeError(f"No valid clips found in: {split_path}")

        print("Training clips:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip = self.build_clips(self.data[idx])
        return clip

    def build_clips(self, item):
        frame_paths, start = item
        W, H = self.target_size

        clip = np.zeros((self.sequence_size, 1, H, W), dtype=np.float32)

        for t in range(self.sequence_size):
            frame_idx = start + t * self.stride
            p = frame_paths[frame_idx]

            with Image.open(p) as img:
                img = img.convert("L").resize((W, H))
                arr = np.array(img, dtype=np.float32) / 255.0   # (H, W)

            clip[t, 0] = arr

        clip = torch.from_numpy(clip).float()
    
        return clip
    
#######------------------------------------------------------#######


#######---------------------Avenue dataset-------------------#######

class AvenueDataset(Dataset):
    def __init__(self, root_dir, split="training_videos", shuffle=True, data_percentage=1.0, sequence_size=10, stride=5, step=5, resize=(256, 256)):
        
        self.root_dir = root_dir
        self.split = split
        self.sequence_size = sequence_size
        self.stride = stride
        self.step = step
        self.resize = resize

        split_dir = os.path.join(root_dir, split)
        self.data = []

        video_files = []
        for dirpath, _, filenames in os.walk(split_dir):
            for fn in filenames:
                if fn.lower().endswith(".avi"):
                    video_files.append(os.path.join(dirpath, fn))

        video_files.sort()

        for video_path in video_files:
            num_frames = self._count_frames(video_path)
            if num_frames <= 0:
                continue

            max_start = num_frames - (self.sequence_size - 1) * self.stride
            if max_start <= 0:
                continue

            for start in range(0, max_start, self.step):
                self.data.append((video_path, start, num_frames))

        if shuffle:
            random.shuffle(self.data)

        limit = int(len(self.data) * data_percentage)
        self.data = self.data[:max(1, limit)]

        if len(self.data) == 0:
            raise RuntimeError(f"No valid clips found in: {split_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.build_clip(self.data[idx])

    def _count_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return num_frames

    def build_clip(self, item):
        video_path, start, _ = item
        W, H = self.resize

        clip = np.zeros((self.sequence_size, 3, H, W), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        for t in range(self.sequence_size):
            frame_id = start + t * self.stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (W, H))
            frame = frame.astype(np.float32) / 255.0
            clip[t] = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        cap.release()
        return torch.from_numpy(clip)  # (T, 3, H, W)    
    
#######---------------------------------------------------#######    
    
    
    
#######---------------------LTD dataset-------------------#######

class LTD_dataset(Dataset):
    def __init__(self, data_split, shuffle=True, data_percentage=1.0, sequence_size=5, stride=5, step=10):
        
        self.data_split = data_split
        root = os.path.join(cfg.LTD_data_path, data_split)

        self.sequence_size = sequence_size
        self.stride = stride
        self.step = step

        self.data = []  # (frame_paths, start)

        for dirpath, _, filenames in os.walk(root):
            frames = [fn for fn in filenames if fn.lower().endswith(".jpg")]
            if not frames:
                continue

            frames = sorted(frames)  # assumes 0001.jpg, 0002.jpg, ...
            frame_paths = [os.path.join(dirpath, fn) for fn in frames]
            n = len(frame_paths)

            max_start = n - (sequence_size - 1) * stride
            if max_start <= 0:
                continue

            for start in range(0, max_start, step):
                self.data.append((frame_paths, start))

        if shuffle:
            random.shuffle(self.data)

        limit = int(len(self.data) * data_percentage)
        self.data = self.data[:max(1, limit)]

        print("Training clips:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip, clip_id = self.build_clips(self.data[idx])
        return clip, clip_id

    def build_clips(self, item):
        frame_paths, start = item

        clip = np.zeros((self.sequence_size, 3, 256, 256), dtype=np.float32)

        for t in range(self.sequence_size):
            p = frame_paths[start + t * self.stride]
            with Image.open(p) as img:
                img = img.convert("RGB").resize((256, 256))
                arr = np.array(img, dtype=np.float32) / 255.0
            clip[t] = arr.transpose(2, 0, 1)

        clip = torch.from_numpy(clip).float()  # (T,3,256,256)
        clip_id = f"{os.path.dirname(frame_paths[0])}|start={start}|stride={self.stride}"
        return clip, clip_id
