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
namespace fs = std::filesystem;
using namespace std;


template <typename T, typename TagT, typename LabelT>
class DebugFriend {
public:
    static void print_internal(const diskann::Index<T, TagT, LabelT>& idx) {
        assert(idx._empty_slots.size() + idx._nd == idx._max_points);
        std::cout << "_nd = " << idx._nd
                  << ", _empty_slots.size() = " << idx._empty_slots.size()
                  << ", _max_points = " << idx._max_points 
                  << ", _is_empty= " << idx._empty_slots.is_empty() << std::endl;
    }

    static void clean_empty_slots(diskann::Index<T, TagT, LabelT>& idx) {
        idx._empty_slots.clear();
        for (size_t i = idx._nd; i < idx._max_points; ++i) {
            idx._empty_slots.insert(i);
        }
    }

    static void batch_search( diskann::Index<T, TagT, LabelT>& idx, uint16_t num_threads,
        T *query,size_t query_aligned_dim,size_t start_query_num,size_t end_query_num,uint32_t recall_at,uint32_t L,float &qps){
        std::vector<uint32_t> query_result_tags(recall_at * (end_query_num - start_query_num));
        std::vector<T *> res = std::vector<T *>(); 
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(static)
        for (int32_t i = start_query_num; i < (int32_t)end_query_num; i++){
            idx.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, false,"" );


        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        qps = (uint32_t)(((end_query_num - start_query_num) / diff.count())/ num_threads);
        cout << "Total time for search: " << diff.count() << " seconds" << std::endl;

    }

};

int main(int argc, char* argv[]){
    diskann::Metric metric = diskann::L2;
    std::string suffix = argv[1];
    float alpha = 1.2f;             
    uint32_t num_threads = 32;  
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
    size_t data_num = 100000;
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

    
    auto index_factory = diskann::IndexFactory(config);

    auto index = index_factory.create_instance();
    auto concrete_index = static_cast<diskann::Index<float, uint32_t>*>(index.get());
    concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
    DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index);
    cout << "Index built and saved successfully." << std::endl;
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    float qps_search;
    float qps_insert;
    float qps_search_baseline;
    float qps_insert_baseline;

    std::string result_file = "../data/batch_search" +suffix+ ".csv";
    std::ofstream result(result_file);
    result << "baseline Qps,num_thread\n";
    for(int i=1; i<=omp_get_num_procs(); i++){
        DebugFriend<float, uint32_t, uint32_t>::batch_search(*concrete_index, i, query, query_aligned_dim, 0, 10000, recall_at, L, qps_search_baseline);
        result << qps_search_baseline << "," << i << "\n";
    }



    return 0;
}
