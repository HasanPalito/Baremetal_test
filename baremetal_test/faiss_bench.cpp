#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <iostream>
#include <vector>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <fcntl.h>
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
    int num = 100000;   
    int M = 32;                
    int ef_construction = 100; 
    int k = 100  ;         // HNSW parameter (number of neighbors in graph)
    // Create the HNSW index (L2 metric)
    faiss::IndexHNSWFlat index(dim, M);
    index.hnsw.efConstruction = 100; 
    std::vector<float> data = read_fbin("../data/sift_query.fbin",query_num,dim);
    //index.add(num, data.data());
    faiss::Index* index_loaded = nullptr;
    try {
        faiss::Index* index = faiss::read_index("../data/HNSW_1M.faiss");
        index_loaded = index;
    } catch (const std::exception& e) {
        std::vector<float> vector_data = read_fbin("../data/sift_0.fbin",num,dim);
        index.add(num, vector_data.data());
        faiss::write_index(&index, "../data/HNSW_1M.faiss");
        std::cout << "Index created and saved to file." << std::endl;
        index_loaded = &index;

    }
    int proc_num = std::stoi(suffix); 
    std::vector<float> latency_stats(query_num, 0);
    
    std::vector<float> D(query_num * k);
    std::vector<faiss::idx_t> I(query_num * k);
    std::string result_file = "../data/batch_search_HNSW" +suffix+ ".csv";
    std::ofstream result(result_file);
    result << "Qps,mean_latencies,num_thread\n";
    
    for(int num_threads=1; num_threads<=omp_get_num_procs(); num_threads++){
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();
            float* D_i = D.data() + i * k;
            faiss::idx_t* I_i = I.data() + i * k;
            index_loaded->search(1, data.data(), k, D_i, I_i);
            auto qe = std::chrono::high_resolution_clock::now();
            latency_stats[i] = std::chrono::duration_cast<std::chrono::microseconds>(qe - qs).count();
            
        }
        auto qps = (uint32_t)(((query_num) / std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - s).count()) / num_threads);
        auto mean_latency =std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);
        result << qps << "," << mean_latency << "," << num_threads << "\n";
    }
    return 0;
}