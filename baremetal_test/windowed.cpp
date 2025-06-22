#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <index.h>
#include <index_factory.h>
#include <stddef.h>
using namespace std;



template <typename T, typename TagT, typename LabelT>
class DebugFriend {
public:
    static std::atomic<bool> worker_done;
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
    static void insert_and_delete( diskann::Index<T, TagT, LabelT>& idx,
        T *query,size_t query_aligned_dim,uint32_t query_num,diskann::IndexWriteParameters &parameters, uint32_t &ips,uint32_t &dps,uint32_t &dips) {
        auto max_point = idx._nd;
        worker_done=false;
        uint32_t count_consolidation = 0;
       
        std::vector<float> ipss;
        std::vector<float> dpss;
        std::vector<float> dipss;
        std::chrono::high_resolution_clock::time_point s;
        for (int32_t i = 1; i < query_num; i++) { 
            if (count_consolidation == 0) {
                s = std::chrono::high_resolution_clock::now();
            }
            count_consolidation++;
            max_point++;
            auto status=idx.insert_point(query + i * query_aligned_dim, max_point);
            if(status !=0){
                cout << "failed insert at " << i << std::endl;
            }
            assert(status == 0);
            idx.lazy_delete(i);
            if (count_consolidation == 10000) {
                std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
                ipss.push_back(count_consolidation / diff.count());
                cout <<count_consolidation / diff.count() << std::endl;
                auto report = idx.consolidate_deletes(parameters);
                dpss.push_back(count_consolidation / report._time);
                std::chrono::duration<double> dips = std::chrono::high_resolution_clock::now() - s;
                dipss.push_back(count_consolidation / dips.count());
                count_consolidation = 0;
            }
        }
        ips = ipss.empty() ? 0.0f : std::accumulate(ipss.begin(), ipss.end(), 0.0f) / ipss.size();
        dps = dpss.empty() ? 0.0f : std::accumulate(dpss.begin(), dpss.end(), 0.0f) / dpss.size();
        dips = dipss.empty() ? 0.0f : std::accumulate(dipss.begin(), dipss.end(), 0.0f) / dipss.size();
        worker_done = true;
    }

    static void simulate_search( diskann::Index<T, TagT, LabelT>& idx, uint16_t num_threads,T *query, size_t query_aligned_dim,uint32_t query_num,uint32_t recall_at,uint32_t L,float &qps){
        std::vector<uint32_t> query_result_tags(recall_at * idx._nd);
        std::vector<T *> res = std::vector<T *>();
        auto s = std::chrono::high_resolution_clock::now();
        uint32_t i = 0;
        omp_set_num_threads(num_threads);
        uint32_t total_query_done= 0;
        while (!worker_done.load()) {
            #pragma omp parallel for schedule(dynamic,1)
            for (uint32_t i=0 ;i<query_num;i++ ){
                idx.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                             query_result_tags.data() + i * recall_at, nullptr, res, false,"" );

            }
            cout <<"total_query:"<<query_num<< std::endl;
            total_query_done = total_query_done + query_num;
            cout <<"total_query:"<<total_query_done<< std::endl;
            
        }
        
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        qps = (uint32_t)((total_query_done / diff.count()) / num_threads);
        cout <<"total_query:"<<total_query_done<< "Total time for search: " << diff.count() << " seconds" << std::endl;
    }

};


template <typename T, typename TagT, typename LabelT>
std::atomic<bool> DebugFriend<T, TagT, LabelT>::worker_done{false};

int main(int argc, char* argv[]){

    diskann::Metric metric = diskann::L2;
    std::string suffix = argv[1];
    float alpha = 1.2f;
    uint32_t num_threads = 32;
    uint32_t R = 32;
    uint32_t L = 100;
    uint32_t build_PQ_bytes = 0;
    bool use_opq = false;
    std::string data_type = "float";
    std::string label_file = "";    
    std::string universal_label = ""; 
    std::string label_type = "uint";
    std::string data_path = "../data/sift_learn.fbin";
    std::string index_path_prefix = "../data/TestIndex/TEST";
    std::string tags_file = "../data/sift_learn.tags";
    std::string query_file = "../data/sift_query.fbin";
    uint32_t data_dim = 128;
    const size_t data_num = 100000;
    bool use_pq_build = false;
    using TagT = uint32_t;
    using T = float;
    using LabelT = uint32_t;

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
                        .with_max_points(data_num+50000 )
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

    std::string result_file  = "../data/windowed" + suffix + ".csv";
    std::ofstream result(result_file);
    result << "num_thread,qps,ips,dps,dips\n";

    
    auto index_factory = diskann::IndexFactory(config);

    auto index = index_factory.create_instance();
    auto concrete_index = static_cast<diskann::Index<float>*>(index.get());
    DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index);
    DebugFriend<float, uint32_t, uint32_t>::print_internal(*concrete_index);

    concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
    DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index);
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(data_path, query, query_num, query_dim, query_aligned_dim);
    T *query_search = nullptr;
    size_t query_num_search, query_dim_search, query_aligned_dim_search, gt_num_search, gt_dim_search;
    diskann::load_aligned_bin<T>(query_file, query_search, query_num_search, query_dim_search, query_aligned_dim_search);
    uint32_t ips, dps,dips;
    for(int i=1; i<=omp_get_num_procs()/2; i++){
        
        auto index = index_factory.create_instance();
        auto concrete_index = static_cast<diskann::Index<float>*>(index.get());
        float qps_search;
        concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
        DebugFriend<float, uint32_t, uint32_t>::clean_empty_slots(*concrete_index);
            //new_concrete_index->load(index_path_prefix.c_str(),4,L);  
        std::thread t1([&, i]() {
            DebugFriend<float, uint32_t, uint32_t>::insert_and_delete(*concrete_index, query, query_aligned_dim, query_num,index_write_params, ips, dps,dips);
        });
        std::thread t2([&, i]() {
            DebugFriend<float, uint32_t, uint32_t>::simulate_search(*concrete_index, i, query_search, query_aligned_dim_search,query_num_search, 10, L, qps_search);
        });
        t1.join();
        t2.join();
        std::cout << "QPS for search with " << i << " threads: " << qps_search << std::endl;
        result << i << "," << qps_search << "," << ips << "," << dps << "," << dips << "\n";
    }
}
