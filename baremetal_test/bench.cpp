#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <index.h>
#include <index_factory.h>
#include <stddef.h>
#include <filesystem>
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
    static std::tuple<uint32_t, std::vector<double>> calculate_recall( diskann::Index<T, TagT, LabelT>& idx,std::string truth_set_file, uint16_t num_threads,
        T *query,size_t query_aligned_dim,size_t query_num,uint32_t recall_at,uint32_t L){
        std::vector<T *> res = std::vector<T *>();    
        uint32_t *gt_ids = nullptr;
        float *gt_dists = nullptr;
        size_t  gt_num, gt_dim;
        
        std::vector<TagT> query_result_tags;
        query_result_tags.resize(recall_at * query_num);
        std::vector<uint32_t> query_result_ids(recall_at * query_num);
        diskann::load_truthset(truth_set_file, gt_ids, gt_dists, gt_num, gt_dim);
        std::vector<double> recalls;
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++){
            auto qs = std::chrono::high_resolution_clock::now();
            
            idx.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, false,"" );

            for (int64_t r = 0; r < (int64_t)recall_at; r++){
                    query_result_ids[recall_at * i + r] = query_result_tags[recall_at * i + r];
            }

            auto qe = std::chrono::high_resolution_clock::now();
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        std::cout << "Total time: " << diff.count() << " seconds" << std::endl;

        for (uint32_t curr_recall = 1; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids.data(), recall_at, curr_recall));
            }
        
        uint32_t qps = (uint32_t)(query_num / diff.count());
        std::cout << "Query per second: " << qps << std::endl;
        std::cout << "Recalls: "<< recalls[99] << std::endl;

        return std::make_tuple(qps, recalls);
    }

};

int main(){
    diskann::Metric metric = diskann::L2;

    float alpha = 1.2f;             
    uint32_t num_threads = 4;  
    uint32_t R = 64;                
    uint32_t L = 100;               
    uint32_t build_PQ_bytes = 0;    
    bool use_opq = false;
    std::string data_type = "float";          
    std::string label_file = "";    
    std::string universal_label = ""; 
    std::string label_type = "uint";
    std::string data_path = "../data/sift_learn.fbin";
    std::string index_path_prefix = "../data/TestIndex/TEST";
    std::string tags_file = "../data/generated_sift_learn.tags";
    std::string truth_set_file= "../data/sift_query_learn_gt100";
    std::string query_file = "../data/sift_query.fbin";
    uint32_t data_dim = 128;
    const size_t data_num = 100000;
    bool use_pq_build = false;
    using TagT = uint32_t;
    using T = float;
    using LabelT = uint32_t;
    std::vector<double> recalls;
    

    uint32_t recall_at = 100;

    // Build index parameters
    auto index_write_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

    auto index_search_params = diskann::IndexSearchParams(L, num_threads);

    // Create the index configuration
    auto config = diskann::IndexConfigBuilder()
                        .with_metric(metric)
                        .with_dimension(data_dim)
                        .with_max_points(data_num + 10)
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
    auto concrete_index = static_cast<diskann::Index<float>*>(index.get());
    concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());

    DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index);
    DebugFriend<float, uint32_t, uint32_t>::print_internal(*concrete_index);
    
    size_t count = 0;

    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;

    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    DebugFriend<float, uint32_t, uint32_t>::calculate_recall(*concrete_index, truth_set_file, num_threads,
                                            query, query_aligned_dim, query_num, recall_at, L);
                                            
    std::string path = "../data/ground_truth";
    int a = 100000;
    std::vector<fs::directory_entry> files;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            files.push_back(entry);
        }
    }
    std::sort(files.begin(), files.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
        return a.path().filename() < b.path().filename();
    });
    std::string result_file = "../data/result.csv";
    std::ofstream file(result_file);
    file << "Num_point,Qps,recall@100\n";
    for (const auto& entry : files) {
        count = 0;
        tsl::robin_set<uint32_t> tags;
        concrete_index->get_active_tags(tags);
        for (auto tag : tags) {
            if (count++ >= a-10000 && count <= a) {
                int status = concrete_index->lazy_delete(tag);
                assert(status == 0);
            }
        }
        concrete_index->consolidate_deletes(index_write_params);
        DebugFriend<float, uint32_t, uint32_t>::print_internal(*concrete_index);
        std::string truth_set_file_reduced= entry.path().string();
        std::cout << "Processing truth set file: " << truth_set_file_reduced << std::endl;
        auto [qps, recalls] =DebugFriend<float, uint32_t, uint32_t>::calculate_recall(*concrete_index, truth_set_file_reduced, num_threads,
                                            query, query_aligned_dim, query_num, recall_at, L);
        a -= 10000;
        file << a << "," << qps << "," << recalls[99] << "\n";
        std::cout << "A: " << a << std::endl;
    }
    file.close();
    //index2 is for the baseline test

    std::string freduced_path= "../data/cleaned";
    std::vector<fs::directory_entry> freduced;
    for (const auto& entry : fs::directory_iterator(freduced_path)) {
        if (entry.is_regular_file()) {
            freduced.push_back(entry); 
        }
    }

    std::sort(freduced.begin(), freduced.end(), [](const auto& a, const auto& b) {
        return a.path().filename() < b.path().filename();
    });

    auto data_num_baseline = data_num - 10000;
    std::string baseline_file = "../data/baseline.csv";
    std::ofstream baseline(baseline_file);

    baseline << "Num_point,Qps,recall@100\n";
    for (size_t i = 0; i < freduced.size(); ++i) {
        
        auto index2= index_factory.create_instance();
        auto concrete_index2 = static_cast<diskann::Index<float>*>(index2.get());
        std::cout << "Pair " << i + 1 << ":\n";
        std::cout << "  " << files[i].path() << "\n";
        std::cout << "  " << freduced[i].path() << "\n";
        concrete_index2->build(freduced[i].path().string().c_str(),  data_num_baseline, tags_file.c_str());
        DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index2);
        auto [qps, recalls] = DebugFriend<float, uint32_t, uint32_t>::calculate_recall(*concrete_index2, files[i].path().string(), num_threads,
                                            query, query_aligned_dim, query_num, recall_at, L);

        baseline << data_num_baseline << "," << qps << "," << recalls[99] << "\n";
        index2.reset();
        data_num_baseline -= 10000;
    }
    file.close();
    return 0;
}