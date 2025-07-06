#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <index.h>
#include <index_factory.h>
#include <stddef.h>
#include <filesystem>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <vector>
#include <fcntl.h>
namespace fs = std::filesystem;
using namespace std;

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


template <typename T, typename TagT, typename LabelT>
class DebugFriend {
public:
    static void brute_search(diskann::Index<T, TagT, LabelT>& idx, const T* query, size_t top_k) {
    using Result = std::pair<float, size_t>;


    auto comp = [](const Result& a, const Result& b) {
        return a.first < b.first; // max-heap: largest distance at top
    };
    std::priority_queue<Result, std::vector<Result>, decltype(comp)> topN(comp);
    assert(query != nullptr);
    for (size_t i = 1; i < idx._nd; ++i) {
        float dist = idx._data_store->get_distance(query, i);
        if (topN.size() < top_k) {
            topN.emplace(dist, i);
        } else if (dist < topN.top().first) {
            topN.pop();
            topN.emplace(dist, i);
        }
    }

    std::vector<Result> result;
    while (!topN.empty()) {
        result.push_back(topN.top());
        topN.pop();
    }

    std::reverse(result.begin(), result.end());

    //for (const auto& [dist, id] : result) {
        //std::cout << "ID: " << id << ", Distance: " << dist << '\n';
    //}
}

};

int main(int argc, char* argv[]){
    diskann::Metric metric = diskann::L2;
    std::string suffix = argv[1];
    float alpha = 1.2f;             
    uint32_t num_threads = 12;  
    uint32_t R = 32;                
    uint32_t L = 100;    
    uint32_t max_L = 350;            
    uint32_t build_PQ_bytes = 0;    
    bool use_opq = false;
    std::string data_type = "float";          
    std::string label_file = "";    
    std::string universal_label = ""; 
    std::string label_type = "uint";
    std::string data_path = "../data/sift_0.fbin";
    std::string index_path_prefix = "../data/TestIndex/TEST";
    std::string tags_file = "../data/tag_for_1m.tags";
    std::string truth_set_file= "../data/1m_point_truth_set";
    std::string query_file = "../data/sift_query.fbin";
    uint32_t data_dim = 128;
    size_t data_num = 1000000;
    bool use_pq_build = false;
    using TagT = uint64_t;
    using T = float;
    using LabelT = uint32_t;
    std::vector<double> recalls;
        
    

    uint32_t recall_at = 10;

    auto index_write_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

    auto index_search_params = diskann::IndexSearchParams(L, num_threads);

    auto config = diskann::IndexConfigBuilder()
                        .with_metric(metric)
                        .with_dimension(data_dim)
                        .with_max_points(data_num + 10000)
                        .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                        .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                        .with_data_type(data_type)
                        .with_label_type(label_type)
                        .is_dynamic_index(true)
                        .with_data_type(diskann_type_to_name<T>())
                        .with_tag_type(diskann_type_to_name<TagT>())
                        .with_label_type(diskann_type_to_name<LabelT>())
                        .with_tag_type(diskann_type_to_name<TagT>())
                        .is_enable_tags(true)
                        .is_use_opq(use_opq)
                        .is_pq_dist_build(use_pq_build)
                        .with_num_pq_chunks(build_PQ_bytes)
                        .with_index_write_params(index_write_params)
                        .with_index_search_params(index_search_params)
                        .build();

    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);
    auto index_factory = diskann::IndexFactory(config);

    auto index = index_factory.create_instance();
    auto concrete_index = static_cast<diskann::Index<float, uint32_t>*>(index.get());
    std::cout << "loading index from: " << index_path_prefix << std::endl;
    try {
        concrete_index->load(index_path_prefix.c_str(), num_threads, L);
    } catch (const std::exception& e) {
        concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
        concrete_index->save(index_path_prefix.c_str(),true);
    }
    std::cout << "succefully loaded" << std::endl;
    
    auto s = std::chrono::high_resolution_clock::now();
    for(int num_threads=1; num_threads<=omp_get_num_procs(); num_threads++){
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < query_num; ++i) {
            cout << "Processing query " << i + 1 << " of " << query_num << std::endl;
            DebugFriend<float, uint32_t, uint32_t>::brute_search(*concrete_index, query + i * query_dim, 100);
        }
    
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        auto qps = (uint32_t)(((query_num) / diff.count()));
        cout << "Total time for search: " << diff.count() << " seconds" << std::endl;
        cout << "Queries per second: " << qps << std::endl;
    }

}