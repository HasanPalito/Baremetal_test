#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <index.h>
#include <index_factory.h>
#include <stddef.h>
using namespace std;

int main(){
    std::cout << "successful" << std::endl;
    
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
    std::string data_path = "data/sift_learn.fbin";
    std::string index_path_prefix = "data/TestIndex/TEST";
    std::string tags_file = "data/sift_learn.tags";
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
                        .with_max_points(data_num+10)
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

    //this will return abstractindex pointer not index
    auto index = index_factory.create_instance();
    auto concrete_index = static_cast<diskann::Index<float>*>(index.get());

    //building index
    concrete_index->build(data_path.c_str(),  data_num, tags_file.c_str());
    //concrete_index->save(index_path_prefix.c_str());

    //inserting dynamicly (with tags)
    vector<float> vector(100);
    for (int i = 0; i < 100; ++i) {
        vector[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float between 0 and 1
    }
    std::cout << "inserting new vector" << std::endl;
    uint32_t mark = 100000;
    auto status = concrete_index->insert_point(vector.data(), mark);
    std::cout << status << std::endl;
    
    std::cout << "deleting new vector" << std::endl;
    auto status_delete= concrete_index->lazy_delete(mark);
    std::cout << status_delete << std::endl;
    //consolidate deletes
    
    std::cout << "searching new vector" << std::endl;
    std::vector<TagT> query_result_tags;
    std::vector<T *> res = std::vector<T *>();
    std::vector<float> distances(10, 0.0f);
    float* distance = distances.data();
    query_result_tags.resize(10);
    
    //example poin
    std::vector<float> sift_vector = {
        0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.1f, 0.9f, 0.4f, 0.7f, 0.5f, 0.6f, 0.8f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.7f, 0.6f, 0.8f, 0.9f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.1f, 0.4f, 0.3f,
        0.6f, 0.9f, 0.8f, 0.2f, 0.1f, 0.6f, 0.5f, 0.7f, 0.8f, 0.3f, 0.9f, 0.4f, 0.6f, 0.1f, 0.7f, 0.2f,
        0.9f, 0.6f, 0.5f, 0.7f, 0.8f, 0.1f, 0.4f, 0.2f, 0.3f, 0.6f, 0.7f, 0.5f, 0.4f, 0.8f, 0.2f, 0.1f,
        0.6f, 0.7f, 0.9f, 0.3f, 0.4f, 0.5f, 0.8f, 0.2f, 0.1f, 0.7f, 0.3f, 0.9f, 0.6f, 0.2f, 0.8f, 0.5f,
        0.4f, 0.9f, 0.6f, 0.3f, 0.8f, 0.2f, 0.4f, 0.7f, 0.5f, 0.6f, 0.9f, 0.1f, 0.4f, 0.8f, 0.6f, 0.7f,
        0.9f, 0.2f, 0.1f, 0.3f, 0.6f, 0.7f, 0.5f, 0.8f, 0.2f, 0.3f, 0.9f, 0.4f, 0.7f, 0.8f, 0.6f, 0.5f
    };

    auto result = concrete_index->search_with_tags(sift_vector.data(), 10, L, query_result_tags.data(), distance,  res);
    for (size_t i = 0; i < query_result_tags.size(); ++i) {
        std::cout << "Tag: " << query_result_tags[i] << "  Distance: " << distances[i] << std::endl;
    }
    
    return 0;
}