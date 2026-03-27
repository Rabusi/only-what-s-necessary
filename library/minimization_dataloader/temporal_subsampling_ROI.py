import os
import cv2
import decord
import decord
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from PIL import Image
from Config import cfg



##########------------------------- UCSD_Ped ---------------------------######

class UCSD_Ped_dataset(Dataset):
    def __init__(self, root_dir, data_split, shuffle=True, data_percentage=1.0, sequence_size=10, stride=5, step=5,
                 mode="full", privacy="raw", blur_radius=4, mask_threshold=25, bg_history=100, bg_var_threshold=16,
                 bg_warmup=20, morph_kernel_size=3):
        
        self.root_dir = root_dir
        self.data_split = data_split
        self.sequence_size = sequence_size
        self.stride = stride
        self.step = step
        self.mode = mode
        self.privacy = privacy

        self.blur_radius = blur_radius
        self.mask_threshold = mask_threshold
        self.bg_history = bg_history
        self.bg_var_threshold = bg_var_threshold
        self.bg_warmup = bg_warmup
        self.morph_kernel_size = morph_kernel_size

        if mode == "full":
            self.target_size = (256, 256)   # (W, H)
        elif mode == "downsamplex2":
            self.target_size = (128, 128)
        elif mode == "downsamplex4":
            self.target_size = (64, 64)
        else:
            raise ValueError("mode must be 'full', 'downsamplex2', or 'downsamplex4'")

        valid_privacy = ["raw", "blur", "masking", "background_removal"]
        if privacy not in valid_privacy:
            raise ValueError(f"privacy must be one of {valid_privacy}")

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

    def _load_frame(self, path, W, H):
        with Image.open(path) as img:
            img = img.convert("L").resize((W, H))
            arr = np.array(img, dtype=np.uint8)   # (H, W)
        return arr

    def _apply_blur(self, gray):
        img = Image.fromarray(gray)
        img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return np.array(img, dtype=np.uint8)

    def _apply_masking(self, gray):
        # Simple threshold masking
        masked = np.where(gray > self.mask_threshold, 255, 0).astype(np.uint8)
        return masked

    def build_clips(self, item):
        frame_paths, start = item
        W, H = self.target_size

        clip = np.zeros((self.sequence_size, 1, H, W), dtype=np.float32)

        backsub = None
        kernel = None

        if self.privacy == "background_removal":
            backsub = cv2.createBackgroundSubtractorMOG2(
                history=self.bg_history,
                varThreshold=self.bg_var_threshold,
                detectShadows=False
            )
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)

            # warm-up
            warmup_start = max(0, start - self.bg_warmup)
            for i in range(warmup_start, start):
                gray = self._load_frame(frame_paths[i], W, H)
                backsub.apply(gray)

        for t in range(self.sequence_size):
            frame_idx = start + t * self.stride
            p = frame_paths[frame_idx]

            gray = self._load_frame(p, W, H)

            if self.privacy == "raw":
                arr = gray

            elif self.privacy == "blur":
                arr = self._apply_blur(gray)

            elif self.privacy == "masking":
                arr = self._apply_masking(gray)

            elif self.privacy == "background_removal":
                fg_mask = backsub.apply(gray)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                arr = cv2.bitwise_and(gray, gray, mask=fg_mask)

            else:
                raise ValueError(f"Unsupported privacy mode: {self.privacy}")

            clip[t, 0] = arr.astype(np.float32) / 255.0

        clip = torch.from_numpy(clip).float()
        return clip
    
#########------------------------------------------------##############    


#########----------------Avenue dataloader---------------##############    

def _sorted_video_paths(video_dir, exts=(".avi",)):
    video_paths = []
    for dirpath, _, filenames in os.walk(video_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                video_paths.append(os.path.join(dirpath, fn))
    video_paths.sort()
    return video_paths


class AvenueDataset(Dataset):
    """
    RGB CUHK Avenue training dataset with median background subtraction.

    Returns:
        clip_tensor: (T, 3, H, W)
        meta: str
    """

    def __init__(self, root_dir, split="training_videos", sequence_size=10, stride=5, step=5, fg_threshold=0.08, bg_samples=30,
                 resize_hw=(256, 256), exts=(".avi",), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalize=False):
        
        self.root_dir = root_dir
        self.split = split
        self.video_dir = os.path.join(root_dir, split)

        self.sequence_size = sequence_size
        self.stride = stride
        self.step = step
        self.fg_threshold = fg_threshold
        self.bg_samples = bg_samples
        self.resize_hw = resize_hw
        self.exts = exts
        self.normalize = normalize

        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

        self.data = []      # list of (video_path, start, n_frames)
        self.bg_cache = {}  # video_path -> background image (H, W, 3)

        video_paths = _sorted_video_paths(self.video_dir, exts=self.exts)

        for video_path in video_paths:
            n_frames = self._get_num_frames(video_path)
            if n_frames <= 0:
                print(f"Skipping {video_path} because n_frames <= 0")
                continue
            max_start = n_frames - (self.sequence_size - 1) * self.stride

            if max_start <= 0:
                print(f"Skipping {video_path} because max_start <= 0")
                continue

            self.bg_cache[video_path] = self._build_bg_rgb(video_path, n_frames)

            for start in range(0, max_start, self.step):
                self.data.append((video_path, start, n_frames))

        if len(self.data) == 0:
            raise RuntimeError(f"No valid Avenue training clips found under: {self.video_dir}")

        print("Avenue training clips:", len(self.data))

    def __len__(self):
        return len(self.data)

    def _get_vr(self, video_path):
        h, w = self.resize_hw
        return decord.VideoReader(video_path, ctx=decord.cpu(), width=w, height=h)

    def _get_num_frames(self, video_path):
        try:
            vr = self._get_vr(video_path)
            return len(vr)
        except Exception as e:
            print(f"Decord failed on {video_path}: {e}")
            return 0

    def _build_bg_rgb(self, video_path, n_frames):
        k = min(self.bg_samples, n_frames)
        idxs = np.linspace(0, n_frames - 1, num=k, dtype=int).tolist()

        vr = self._get_vr(video_path)

        try:
            frames = vr.get_batch(idxs).asnumpy()  # (K, H, W, 3), RGB
        except Exception as e:
            raise RuntimeError(f"Could not sample frames from: {video_path}. Error: {e}")

        frames = frames.astype(np.float32) / 255.0
        bg = np.median(frames, axis=0).astype(np.float32)
        return bg

    def _normalize_clip(self, clip):
        # clip: (T, 3, H, W)
        return (clip - self.mean) / self.std

    def __getitem__(self, idx):
        video_path, start, n_frames = self.data[idx]
        bg = self.bg_cache[video_path]
        H, W = self.resize_hw

        frame_indices = [start + t * self.stride for t in range(self.sequence_size)]

        if frame_indices[-1] >= n_frames:
            raise RuntimeError(f"Requested frame {frame_indices[-1]} but video has only {n_frames} frames: {video_path}")

        vr = self._get_vr(video_path)

        try:
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, 3), RGB
        except Exception as e:
            raise RuntimeError(f"Could not read frames {frame_indices} from {video_path}. Error: {e}")

        frames = frames.astype(np.float32) / 255.0
        clip = np.zeros((self.sequence_size, 3, H, W), dtype=np.float32)

        for t in range(self.sequence_size):
            rgb = frames[t]  # (H, W, 3)

            diff = np.abs(rgb - bg)              # (H, W, 3)
            diff_scalar = np.mean(diff, axis=2)  # (H, W)
            mask = (diff_scalar > self.fg_threshold).astype(np.float32)

            fg = rgb * mask[..., None]           # (H, W, 3)
            clip[t] = np.transpose(fg, (2, 0, 1))  # (3, H, W)

        if self.normalize:
            clip = self._normalize_clip(clip)

        clip_tensor = torch.from_numpy(clip).float()
        meta = f"{video_path}|start={start}|stride={self.stride}"

        return clip_tensor, meta
