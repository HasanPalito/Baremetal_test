#include "hnswlib/hnswlib/hnswlib.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <index.h>
#include <index_factory.h>
#include <stddef.h>
#include <filesystem>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <fcntl.h>

long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                     int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}


int open_perf_event(perf_event_attr &pe) {
    return perf_event_open(&pe, 0, -1, -1, 0);
}

int create_perf_event(uint32_t type, uint64_t config) {
    struct perf_event_attr pe{};
    std::memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return open_perf_event(pe);
}


std::vector<float> read_fbin(const std::string& filename, int& num, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + filename);

    input.read(reinterpret_cast<char*>(&num), 4);
    input.read(reinterpret_cast<char*>(&dim), 4);

    std::vector<float> data(num * dim);
    input.read(reinterpret_cast<char*>(data.data()), num * dim * 4);
    if (input.gcount() != num * dim * 4)
        throw std::runtime_error("File ended unexpectedly!");

    return data;
}

int main(int argc, char* argv[]) {
    std::string suffix = argv[1];
    int dim = 128;             
    int query_num = 10000;  
    int num = 1000000;   
    int M = 32;                
    int ef_construction = 100; 

    // Initing index
    hnswlib::L2Space space(dim);
    std::vector<float> data = read_fbin("../data/sift_query.fbin", query_num, dim);
    std::vector<float> latency_stats(query_num, 0);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_loaded;
    try {
        index_loaded = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, "../data/hnsw_index_1M.bin");
    } catch (const std::exception& e) {
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num, M, ef_construction);
        for (int i = 0; i < num; i++) {
            alg_hnsw->addPoint(data.data() + i * dim, i); 
        }
        alg_hnsw->saveIndex("../data/hnsw_index_1M.bin");
        index_loaded = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, "../data/hnsw_index_1M.bin");
    }
    std::string result_file = "../data/batch_search_HNSW" +suffix+ ".csv";
    std::ofstream result(result_file);
    result << "Qps,mean_latencies,num_thread\n";

    int proc_num = std::stoi(suffix);
    for (int num_threads = 1; num_threads <= omp_get_num_procs(); num_threads++) {
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = index_loaded->searchKnn(data.data() + i * dim, 1);
            auto qe = std::chrono::high_resolution_clock::now();
            latency_stats[i] = std::chrono::duration_cast<std::chrono::microseconds>(qe - qs).count();
        }
        auto e = std::chrono::high_resolution_clock::now();
        auto qps = (uint32_t)(((query_num) / std::chrono::duration_cast<std::chrono::seconds>(e - s).count()) / num_threads);
        auto mean_latency =std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);
        result << qps << "," << mean_latency << "," << num_threads << "\n";
    }
    return 0;
}