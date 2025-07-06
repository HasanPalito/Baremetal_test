#!/bin/bash

echo "args,cache_misses,branch_misses,context_switches,cpu_migrations,instruction_per_cycle" > HNSW_FAISS.csv

for file in HNSW_faiss_lib_*.csv; do
    # Extract numeric argument from filename (e.g., result10.csv → 10)
    arg=$(echo "$file" | grep -oP '\d+')

    # Extract values
    cache_misses=$(grep '^.*cache-misses' "$file" | cut -d',' -f1)
    branch_misses=$(grep '^.*branch-misses' "$file" | cut -d',' -f1)
    context_switches=$(grep '^.*context-switches' "$file" | cut -d',' -f1)
    cpu_migrations=$(grep '^.*cpu-migrations' "$file" | cut -d',' -f1)
    ipc=$(grep '^.*instructions' "$file" | awk -F',' '{print $(NF-1)}')

    # Write row
    echo "$arg,$cache_misses,$branch_misses,$context_switches,$cpu_migrations,$ipc" >> HNSW_FAISS.csv
done

# Sort merged by args (numerically)
sort -n HNSW_FAISS.csv -o HNSW_FAISS.csv