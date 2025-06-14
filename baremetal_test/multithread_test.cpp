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

uint64_t generate_uint64_from_uuid() {
    boost::uuids::uuid unique_id = boost::uuids::random_generator()();
    
    uint64_t first_half = *reinterpret_cast<const uint64_t*>(&unique_id.data[0]);
    uint64_t second_half = *reinterpret_cast<const uint64_t*>(&unique_id.data[8]);
    return first_half ^ second_half;
}

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
        std::vector<uint64_t> query_result_tags(recall_at * (end_query_num - start_query_num));
        std::vector<T *> res = std::vector<T *>(); 
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = start_query_num; i < (int64_t)end_query_num; i++){
            idx.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, false,"" );


        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        qps = (uint32_t)(((end_query_num - start_query_num) / diff.count())/ num_threads);
        cout << "Total time for search: " << diff.count() << " seconds" << std::endl;

    }
    static void batch_insert( diskann::Index<T, TagT, LabelT>& idx, uint16_t num_threads,
        T *query,size_t query_aligned_dim,size_t start_query_num,size_t end_query_num,uint32_t recall_at,uint32_t L,float &qps){
        std::vector<uint64_t> query_result_tags(recall_at * (end_query_num - start_query_num));
        std::vector<T *> res = std::vector<T *>(); 
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = start_query_num; i < (int64_t)end_query_num; i++){
            //auto timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            auto status=idx.insert_point(query + i * query_aligned_dim, generate_uint64_from_uuid());
            if(status !=0){
                cout << "failed insert at " << i-start_query_num  << std::endl;
            }
            assert(status == 0);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        qps = (uint32_t)(((end_query_num - start_query_num) / diff.count())/ num_threads);
        cout << "Total time for insert: " << diff.count() << " seconds" << std::endl;
    }


};

int main(){
    diskann::Metric metric = diskann::L2;

    float alpha = 1.2f;             
    uint32_t num_threads = 8;  
    uint32_t R = 8;                
    uint32_t L = 10;    
    uint32_t max_L = 350;            
    uint32_t build_PQ_bytes = 0;    
    bool use_opq = false;
    std::string data_type = "float";          
    std::string label_file = "";    
    std::string universal_label = ""; 
    std::string label_type = "uint";
    std::string data_path = "../data/sift_learn.fbin";
    std::string index_path_prefix = "../data/TestIndex/TEST";
    std::string tags_file = "../data/sift_64.tags";
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
    auto concrete_index = static_cast<diskann::Index<float, uint64_t>*>(index.get());
    concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
    DebugFriend<float, uint64_t, uint32_t>::clean_empty_slots(*concrete_index);
    concrete_index->save(index_path_prefix.c_str(),true);
    cout << "Index built and saved successfully." << std::endl;
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    float qps_search;
    float qps_insert;
    float qps_search_baseline;
    float qps_insert_baseline;

    std::string data_dir= "../data/cleaned";
    std::vector<fs::directory_entry> data_files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            data_files.push_back(entry);
        }
    }

    std::sort(data_files.begin(), data_files.end(), [](const auto& a, const auto& b) {
        return a.path().filename() < b.path().filename();
    });

    std::string result_file = "../data/multithreaded.csv";
    std::ofstream result(result_file);
    result << "Num_point,concurent Qps insert,concurent Qps search,baseline Qps insert,baseline Qps search,num_thread\n";

    DebugFriend<float, uint64_t, uint32_t>::print_internal(*concrete_index);
    for (const auto& entry : data_files) {
        std::cout << "Data file: " << entry.path() << std::endl;
        for(int i=1; i<=omp_get_num_procs()/2; i++){
            auto new_index = index_factory.create_instance();
            auto new_concrete_index = static_cast<diskann::Index<float, uint64_t, uint32_t>*>(new_index.get());
            std::cout << "Loading index for thread " << i << std::endl;
            new_concrete_index->build(entry.path().string().c_str(),  data_num, tags_file.c_str());
            //new_concrete_index->load(index_path_prefix.c_str(),4,L);
            DebugFriend<float, uint64_t, uint32_t>::clean_empty_slots(*new_concrete_index);
            DebugFriend<float, uint64_t, uint32_t>::print_internal(*new_concrete_index);
            std::thread t1([&, i]() {
                DebugFriend<float, uint64_t, uint32_t>::batch_search(*new_concrete_index, i, query, query_aligned_dim, 0, 5000, recall_at, L, qps_search);
            });
            std::thread t2([&, i]() {
                DebugFriend<float, uint64_t, uint32_t>::batch_insert(*new_concrete_index, i, query, query_aligned_dim, 5000, 10000, recall_at, L, qps_insert);
            });
            t1.join();
            t2.join();
            std::cout << "QPS Search: " << qps_search << ", QPS Insert: " << qps_insert << " num_thread: " << i << std::endl;
            auto new_baseline_index = index_factory.create_instance();
            auto new_baseline_concrete_index = static_cast<diskann::Index<float, uint64_t, uint32_t>*>(new_baseline_index.get());
            std::cout << "Loading index for thread " << i << std::endl;
            new_baseline_concrete_index->build(entry.path().string().c_str(),  data_num, tags_file.c_str());
            DebugFriend<float, uint64_t, uint32_t>::clean_empty_slots(*new_baseline_concrete_index);
            DebugFriend<float, uint64_t, uint32_t>::print_internal(*new_baseline_concrete_index);
            DebugFriend<float, uint64_t, uint32_t>::batch_search(*new_baseline_concrete_index, i, query, query_aligned_dim, 0, 5000, recall_at, L, qps_search_baseline);
            DebugFriend<float, uint64_t, uint32_t>::batch_insert(*new_baseline_concrete_index, i, query, query_aligned_dim, 5000, 10000, recall_at, L, qps_insert_baseline);
            std::cout << "QPS Search: " << qps_search_baseline << ", QPS Insert: " << qps_insert_baseline << " num_thread: " << i << " baseline" << std::endl;
            result << data_num << "," << qps_insert << "," << qps_search << "," << qps_insert_baseline << "," << qps_search_baseline << "," << i << "\n";
        }

        data_num -= 10000; 
    }


    return 0;
}
