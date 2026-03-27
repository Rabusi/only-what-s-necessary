import os, sys
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Config import cfg




ATTRS = ["skin_color", "relationship", "face", "nudity", "gender"]


def segment_value_at_t(segments, t, default=0):
    for s, e, v in segments:
        if s <= t <= e:
            return v
    return default


def binarize_privacy(attr, val):
    if isinstance(val, (list, tuple)):
        return 1

    if attr == "skin_color":
        return 0 if val == 0 else 1
    if attr == "face":
        return 1 if val in (1, 2) else 0
    if attr == "gender":
        return 0 if val == 0 else 1
    if attr == "nudity":
        return 1 if val in (1, 2) else 0
    if attr == "relationship":
        return 1 if val == 1 else 0

    raise ValueError(f"Unknown attribute: {attr}")


def load_all_annotations(privacy_json_dir):
    ann_map = {}
    for fn in sorted(os.listdir(privacy_json_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(privacy_json_dir, fn)
        with open(path, "r") as f:
            data = json.load(f)
        for video_name, ann in data.items():
            ann_map[video_name] = ann
    return ann_map


def read_video_rgb(video_path, size=(256, 256)):
    frames = []
    reader = imageio.get_reader(video_path, "ffmpeg")
    for frame in reader:
        img = Image.fromarray(frame).convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        frames.append(arr)
    reader.close()
    x = np.stack(frames, axis=0)          # (T,H,W,3)
    x = np.transpose(x, (0, 3, 1, 2))     # (T,3,H,W)
    return x


def build_privacy_labels_per_frame(ann_dict, T):
    labels = np.zeros((T, len(ATTRS)), dtype=np.int64)
    for t in range(T):
        for j, a in enumerate(ATTRS):
            segs = ann_dict.get(a, [])
            val = segment_value_at_t(segs, t, default=0)
            labels[t, j] = binarize_privacy(a, val)
    return labels  # (T, 5)


def split_videos(video_names, train_ratio=0.6, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(video_names))
    shuffled = [video_names[i] for i in perm]
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]



class PAHMDB_Privacy_Dataset(Dataset):
    """
    Contiguous non-overlapping clips:
      clip k => frames [k*SEQ_LEN : (k+1)*SEQ_LEN]
    Drops the last incomplete chunk.
    """

    def __init__(self, ann_map, idx_map, video_names, seq_len=10, size=(256, 256)):
        """
        ann_map: dict {basename.avi -> annotation_dict}
        idx_map: dict {basename.avi -> full_path}
        video_names: list of basenames (must be keys in ann_map/idx_map)
        """
        self.ann_map = ann_map
        self.idx_map = idx_map
        self.video_names = video_names
        self.seq_len = seq_len
        self.size = size

        self.index = []  # list of (video_name, clip_start)
        for vn in self.video_names:
            if vn not in self.ann_map:
                continue
            if vn not in self.idx_map:
                continue

            vp = self.idx_map[vn]
            if not os.path.isfile(vp):
                continue

            reader = imageio.get_reader(vp, "ffmpeg")
            try:
                T = reader.count_frames()
            except Exception:
                T = 0
                for _ in reader:
                    T += 1
            reader.close()

            n_clips = T // self.seq_len
            for k in range(n_clips):
                self.index.append((vn, k * self.seq_len))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vn, start = self.index[idx]
        vp = self.idx_map[vn]

        frames = read_video_rgb(vp, size=self.size)   # (T,H,W)
        labels = build_privacy_labels_per_frame(self.ann_map[vn], T=len(frames))  # (T,5)

        clip_frames = frames[start:start + self.seq_len]    # (seq,H,W)
        clip_labels = labels[start:start + self.seq_len]    # (seq,5)

        clips = torch.from_numpy(clip_frames).float()       # (seq,3,H,W)
        priv_label = torch.from_numpy(clip_labels).long()   # (seq,5)
        
        return clips, priv_label
    

def build_basename_to_path(video_root):
    idx = {}
    for cls in os.listdir(video_root):
        cls_dir = os.path.join(video_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(".avi"):
                idx[fn] = os.path.join(cls_dir, fn)
    return idx


def video_paths(video_root, ann_map):
    ann_keys = set(ann_map.keys())
    paths = []
    for class_name in os.listdir(video_root):
        class_dir = os.path.join(video_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fn in os.listdir(class_dir):
            if fn.lower().endswith(".avi") and fn in ann_keys:
                paths.append(os.path.join(class_dir, fn))
    paths.sort()
    return paths


def build_loaders(video_dir, privacy_json_dir, seq_len=10, size=(256, 256), train_ratio=0.6, seed=42):

    ann_map = load_all_annotations(privacy_json_dir)
    print(f"Loaded annotations for {len(ann_map)} videos.")

    idx = build_basename_to_path(video_dir)

    all_videos = sorted([v for v in ann_map.keys() if v in idx])
    print("usable:", len(all_videos), "annotations:", len(ann_map))

    train_videos, test_videos = split_videos(all_videos, train_ratio=train_ratio, seed=seed)

    train_ds = PAHMDB_Privacy_Dataset(ann_map, idx, train_videos, seq_len=seq_len, size=size)
    test_ds  = PAHMDB_Privacy_Dataset(ann_map, idx, test_videos,  seq_len=seq_len, size=size)

    return train_ds, test_ds


def check_binary(train_loader, n_attrs=5, n_batches=10):
    vals = [set() for _ in range(n_attrs)]
    for i, (_, y) in enumerate(train_loader):
        y = y.view(-1, n_attrs).cpu()
        for j in range(n_attrs):
            vals[j].update(y[:, j].unique().tolist())
        if i + 1 >= n_batches:
            break
    return vals


if __name__ == "__main__":
    VIDEO_DIR = cfg.PAHMDB_data_path
    PRIV_DIR  = cfg.PAHMDB_privacy_json_dir

    train_loader, test_loader = build_loaders(VIDEO_DIR, PRIV_DIR, seq_len=10, size=(256, 256), train_ratio=0.6, seed=42, batch_size=4, num_workers=2)

    train_loader = DataLoader(train_loader, batch_size=4, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_loader,  batch_size=4, shuffle=False, num_workers=2)

    vals = check_binary(train_loader)
    print(vals)
    
    clips, priv_label = next(iter(train_loader))
    print(f"clip shape: {clips.shape}")  # (B, 10, 1, 256, 256)
    print(f"privacy label shape: {priv_label.shape}")  # (B, 10, 5)