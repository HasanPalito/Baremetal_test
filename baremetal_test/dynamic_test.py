from diskannpy import DynamicMemoryIndex
import utils as ut
import diskannpy
import numpy as np
import utils
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

index.from_file(
    "sift_3",
    vector_dtype="float32",
    distance_metric="l2",
    complexity=64,
    graph_degree=32,
    num_threads=8,
    max_vectors=1000000,
)

len= utils.read_tags_length("sift_3/ann.tags")

num_vectors = 100
dim = 128
vectors = np.random.rand(num_vectors, dim).astype(np.float32)
tags = np.arange(len+1, len + num_vectors+1, dtype=np.uint32)

print("inserting")
for i in range(num_vectors):
    vector = vectors[i]            
    tag = tags[i]
    index.insert(vector, tag) 

print("deleting")
for i in range(num_vectors):           
    tag = tags[i]
    index.mark_deleted(tag)

result = index.search(vectors[0], 10, complexity)  
print(result)

print("hard deleting")
index.consolidate_delete()



