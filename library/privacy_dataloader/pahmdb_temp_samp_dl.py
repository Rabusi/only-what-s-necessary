import os, json
import sys
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


def count_frames_fast(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        T = reader.count_frames()
    except Exception:
        T = 0
        for _ in reader:
            T += 1
    reader.close()
    return T


def read_frames_at_indices(video_path, indices, size=(256, 256)):
    indices = list(map(int, indices))
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for i in indices:
        frame = reader.get_data(i)
        img = Image.fromarray(frame).convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        frames.append(arr)
    reader.close()
    x = np.stack(frames, axis=0)          # (seq,H,W,3)
    x = np.transpose(x, (0, 3, 1, 2))     # (seq,3,H,W)
    return x


def privacy_labels_for_indices(ann_dict, indices):
    labels = np.zeros((len(indices), len(ATTRS)), dtype=np.int64)
    for k, t in enumerate(indices):
        for j, a in enumerate(ATTRS):
            segs = ann_dict.get(a, [])
            val = segment_value_at_t(segs, t, default=0)
            labels[k, j] = binarize_privacy(a, val)
    return labels


def split_videos(video_names, train_ratio=0.6, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(video_names))
    shuffled = [video_names[i] for i in perm]
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


########--------- Deterministic Sampling Dataset ---------########    

class PAHMDB_temporal_sampling(Dataset):
    """
    Deterministic non-overlapping temporal subsampling clips.
    
    definition of stride:
      stride=1 means skip 1 frame  -> step = 2
      stride=2 means skip 2 frames -> step = 3

    Frames in a clip (1-based frame numbers):
      start1, start1+step, ..., start1+(seq_len-1)*step

    Starts "and so on":
      start1 = offset_1based, offset_1based + hop, offset_1based + 2*hop, ...
    where hop = seq_len * step
    """

    def __init__(self, ann_map, idx_map, video_names, seq_len=10, stride=2, offset_1based=1, size=(256, 256)):

        self.ann_map = ann_map
        self.idx_map = idx_map
        self.video_names = [vn for vn in video_names if vn in ann_map and vn in idx_map]
        self.seq_len = seq_len
        self.stride = stride
        self.step = self.stride + 1
        self.offset_1based = offset_1based
        self.size = size

        self.index = []  
        hop = self.seq_len * self.step

        for vn in self.video_names:
            vp = self.idx_map[vn]
            if not os.path.isfile(vp):
                continue

            T = count_frames_fast(vp)
            if T <= 0:
                continue
            max_start0 = T - 1 - (self.seq_len - 1) * self.step
            if max_start0 < 0:
                continue

            for start1 in range(self.offset_1based, (max_start0 + 1) + 1, hop):
                start0 = start1 - 1 
                self.index.append((vn, start0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vn, start0 = self.index[idx]
        vp = self.idx_map[vn]

        indices0 = [start0 + k * self.step for k in range(self.seq_len)]
        frames = read_frames_at_indices(vp, indices0, size=self.size)  # (seq,3,H,W)

        indices1 = [i + 1 for i in indices0]
        labels = privacy_labels_for_indices(self.ann_map[vn], indices1)  # (seq,5)

        clips = torch.from_numpy(frames).float()       # (seq,3,H,W)
        priv_label = torch.from_numpy(labels).long()   # (seq,5)
        return clips, priv_label
    
    
def build_loaders(VIDEO_DIR, PRIV_DIR, seq_len=10, stride=2, size=(256, 256), train_ratio=0.6, seed=42):
    ann_map = load_all_annotations(PRIV_DIR)
    idx_map = build_basename_to_path(VIDEO_DIR)

    all_videos = sorted([v for v in ann_map.keys() if v in idx_map])
    train_videos, test_videos = split_videos(all_videos, train_ratio=train_ratio, seed=seed)

    # Experiment A (offset 1, frame start from 1)
    train_ds_A = PAHMDB_temporal_sampling(ann_map, idx_map, train_videos, seq_len=seq_len, stride=stride, offset_1based=1, size=size)
    test_ds_A = PAHMDB_temporal_sampling(ann_map, idx_map, test_videos,  seq_len=seq_len, stride=stride, offset_1based=1, size=size)

    # Experiment B (offset 2, frame start from 2)
    # train_ds_B = PAHMDB_temporal_sampling(ann_map, idx_map, train_videos, seq_len=seq_len, stride=stride, offset_1based=2, size=size)
    # test_ds_B = PAHMDB_temporal_sampling(ann_map, idx_map, test_videos,  seq_len=seq_len, stride=stride, offset_1based=2, size=size)

    return train_ds_A, test_ds_A


########--------- Random Sampling Dataset ---------########    
    
class PAHMDB_random_temporal_sampling(Dataset):
    """
    Random temporal subsampling clips.

    Your meaning:
      stride = number of frames to SKIP
      step   = stride + 1

    Clip frames (1-based):
      start1, start1+step, ..., start1+(seq_len-1)*step

    start1 is chosen randomly (no phase / offset constraint).
    """

    def __init__(self, ann_map, idx_map, video_names, seq_len=10, stride=1, size=(256, 256), seed=42, samples_per_video=10):
        self.ann_map = ann_map
        self.idx_map = idx_map
        self.video_names = [vn for vn in video_names if vn in ann_map and vn in idx_map]
        self.seq_len = seq_len
        self.epoch = 0

        self.stride = stride     # skip
        self.step = self.stride + 1   # sampling step

        self.size = size
        self.seed = seed
        self.samples_per_video = samples_per_video

        # store per-video max start (0-based)
        self.video_info = []  # list of (vn, max_start0)
        for vn in self.video_names:
            vp = self.idx_map[vn]
            if not os.path.isfile(vp):
                continue
            T = count_frames_fast(vp)
            if T <= 0:
                continue

            max_start0 = T - 1 - (self.seq_len - 1) * self.step
            if max_start0 >= 0:
                self.video_info.append((vn, max_start0))

        # expand index: samples_per_video random clips per video
        self.index = []
        for vn, max_start0 in self.video_info:
            for _ in range(self.samples_per_video):
                self.index.append((vn, max_start0))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vn, max_start0 = self.index[idx]
        vp = self.idx_map[vn]

        rng = np.random.default_rng(self.seed + 100000 * self.epoch + idx)
        start0 = int(rng.integers(0, max_start0 + 1))  # 0-based

        indices0 = [start0 + k * self.step for k in range(self.seq_len)]
        frames = read_frames_at_indices(vp, indices0, size=self.size)  # (seq,3,H,W)

        indices1 = [i + 1 for i in indices0]  # labels are 1-based
        labels = privacy_labels_for_indices(self.ann_map[vn], indices1)  # (seq,5)

        clips = torch.from_numpy(frames).float()
        priv_label = torch.from_numpy(labels).long()
        return clips, priv_label
    


def build_random_loader(VIDEO_DIR, PRIV_DIR, seq_len=10, stride=1, size=(256, 256), train_ratio=0.6, seed=42, samples_per_video=10):
    ann_map = load_all_annotations(PRIV_DIR)
    idx_map = build_basename_to_path(VIDEO_DIR)

    all_videos = sorted([v for v in ann_map.keys() if v in idx_map])
    train_videos, test_videos = split_videos(all_videos, train_ratio=train_ratio, seed=seed)

    train_ds_A = PAHMDB_random_temporal_sampling(ann_map, 
                                                 idx_map, 
                                                 train_videos, seq_len=seq_len, 
                                                 stride=stride, 
                                                 size=size, 
                                                 seed=seed, 
                                                 samples_per_video=samples_per_video)
    test_ds_A = PAHMDB_random_temporal_sampling(ann_map, 
                                                idx_map, 
                                                test_videos, 
                                                seq_len=seq_len, 
                                                stride=stride, 
                                                size=size, 
                                                seed=seed, 
                                                samples_per_video=samples_per_video)

    # train_ds_B = PAHMDB_random_temporal_sampling(ann_map, idx_map, train_videos,seq_len=seq_len, stride=stride, size=size,seed=seed + 1, samples_per_video=samples_per_video)
    # test_ds_B = PAHMDB_random_temporal_sampling(ann_map, idx_map, test_videos,seq_len=seq_len, stride=stride, size=size,seed=seed + 1, samples_per_video=samples_per_video)


    return train_ds_A, test_ds_A



#######---------main--------########

if __name__ == "__main__":
    VIDEO_DIR = cfg.PAHMDB_data_path
    PRIV_DIR  = cfg.PAHMDB_privacy_json_dir

    # train_ds_A, test_ds_A = build_loaders(VIDEO_DIR, PRIV_DIR, 
    #                                       seq_len=10, stride=1, size=(256,256), 
    #                                       train_ratio=0.6, seed=42,
    #                                       batch_size=4, num_workers=2)
    
    train_ds_A, test_ds_A = build_random_loader(VIDEO_DIR, 
                                                PRIV_DIR, 
                                                seq_len=10, 
                                                stride=1, 
                                                size=(256,256), 
                                                train_ratio=0.6, 
                                                seed=42, 
                                                samples_per_video=10)    

    print(len(train_ds_A), len(test_ds_A))
    
    train_loader_A = DataLoader(train_ds_A, batch_size=4, shuffle=True,  num_workers=2)
    test_loader_A  = DataLoader(test_ds_A,  batch_size=4, shuffle=False, num_workers=2)

    # train_loader_B = DataLoader(train_ds_B, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    # test_loader_B  = DataLoader(test_ds_B,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

    # use loaders directly (do NOT wrap again)
    clips, priv_label = next(iter(train_loader_A))
    print("clips:", clips.shape)        # (B, seq, 3, H, W)
    print("priv:", priv_label.shape)    # (B, seq, 5)
    
    
    # clips, priv_label = next(iter(train_loader_B))
    # print("clips:", clips.shape)        # (B, seq, 3, H, W)
    # print("priv:", priv_label.shape)    # (B, seq, 5)
