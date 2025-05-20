from diskannpy import DynamicMemoryIndex
import utils as ut
import diskannpy
import numpy as np

dimensions = 128         # Dimensionality of your vectors
max_vectors = 1_000_000  # Maximum capacity of the index
complexity = 64          # Trade-off between indexing time and quality (64 is typical)
graph_degree = 32        # Degree of the graph (usually 32 is a good default)
metric = "l2"            # or "cosine"

index = DynamicMemoryIndex(
    vector_dtype="float32",
    dimensions=dimensions,
    max_vectors=max_vectors,
    complexity=complexity,
    graph_degree=graph_degree,
    distance_metric=metric,
)

def load_diskann_bin(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(n, d)
        return data
    
data = load_diskann_bin("python_data/sift_learn.fbin")


def write_tags_bin(tags: np.ndarray, path: str):
    num_points = tags.shape[0]
    header = np.array([num_points, 1], dtype=np.uint32)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tags.astype(np.uint32).tobytes())

tags = np.arange(len(data), dtype=np.uint32)
write_tags_bin(tags, "ann.tags")

diskannpy.build_memory_index(
    data=data,
    index_directory="sift_3",
    vector_dtype="float32",
    distance_metric="l2",           
    complexity=64,                  
    graph_degree=32,                 
    num_threads=8,
    tags="ann.tags"             
)

index.from_file(
    "sift_3",
    vector_dtype="float32",
    distance_metric="l2",
    complexity=64,
    graph_degree=32,
    num_threads=8,
    max_vectors=1000000,
)




