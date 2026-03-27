import random
import numpy as np
from os import listdir
from os.path import isdir, join
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import numpy as np
from torch.utils.data import Dataset


import os
import numpy as np
from PIL import Image

from Config import cfg


#####----------------- GET CLIPS BY STRIDE -----------------#####

def get_clips_by_stride(stride, frames_list, sequence_size):
    """
    Create non-overlapping clips of length `sequence_size` using the given stride.
    For stride=2, it creates two streams: start=0 and start=1.
    """
    clips = []
    sz = len(frames_list)

    for start in range(stride):
        clip = np.zeros((sequence_size, 256, 256, 1), dtype=np.float32)
        cnt = 0

        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt += 1

            if cnt == sequence_size:
                clips.append(clip.copy())
                cnt = 0
                clip.fill(0.0)

    return clips

############-------------------------------------------############


######----------------- GET TRAINING SET -----------------#########

def get_training_set(mode="full"):
    """
    Args:
        mode:
            'full' -> keep 256x256
            'downsamplex2' -> downsamplex2 -> 128x128
            'downsamplex4' -> downsamplex4 -> 64x64
    Returns:
        list of clips
    """

    if mode == "full":
        target_size = (256, 256)
    elif mode == "downsamplex2":
        target_size = (128, 128)
    elif mode == "downsamplex4":
        target_size = (64, 64)
    else:
        raise ValueError("mode must be 'full', 'downsamplex2', or 'downsamplex4'")
    
    clips = []

    for f in sorted(listdir(cfg.DATASET_PATH)):
        directory_path = join(cfg.DATASET_PATH, f)

        all_frames = []

        for c in sorted(listdir(directory_path)):
            img_path = join(directory_path, c)
            if not img_path.lower().endswith(".tif"):
                continue

            with Image.open(img_path) as img:
                img = img.resize(target_size)
                frame = np.array(img, dtype=np.float32) / 255.0

                if frame.ndim == 2:
                    frame = frame[..., None]   # (H, W) -> (H, W, 1)

            all_frames.append(frame)

        if len(all_frames) < 10:
            continue

        for stride in range(1, 3):
            clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))

    return clips

############-------------------------------------------############


#####----------------- GET SINGLE TEST SET -----------------#####

def get_single_test(test_path):
    frames = []
    prev_img = None

    for f in sorted(listdir(test_path)):
        file_path = join(test_path, f)
        if not file_path.lower().endswith((".tif", ".tiff")):
            continue
        try:
            with Image.open(file_path) as img:
                img = img.convert("L")                  
                img = img.resize((256, 256))
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)    

        except Exception as e:
            print(f"[WARNING] Skipping bad frame {file_path}: {e}")

            if prev_img is not None:
                img = prev_img.copy()
            else:
                img = np.zeros((256, 256, 1), dtype=np.float32)

        frames.append(img)
        prev_img = img

    test = np.array(frames, dtype=np.float32)          
    return test

############-------------------------------------------############