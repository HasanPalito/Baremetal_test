import numpy as np
import os

def load_diskann_bin(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(n, d)
        return data
    
def write_tags_bin(tags: np.ndarray, path: str):
    num_points = tags.shape[0]
    header = np.array([num_points, 1], dtype=np.uint32)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tags.astype(np.uint32).tobytes())

def read_tags_length(tags_path: str) -> int:
    with open(tags_path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        num_tags, dim = header
        if dim != 1:
            raise ValueError(f"Invalid tag dimension: expected 1, got {dim}")
        return num_tags