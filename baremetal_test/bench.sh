#!/bin/bash



for i in $(seq 1 3); do
    ./bench_search_2 4 $i
    ./bench_search_2 8 $i
    ./bench_search_2 16 $i
    ./bench_search_2 32 $i
    ./bench_search_2 64 $i
    ./bench_search  $i
    ./fais_search $i
    ./hnsw_bench $i
    ./brute_search $i

done