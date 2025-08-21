#include "genpriors.h"

void priorGen(std::vector<std::vector<float>> &priors, cv::Size inputSize){
    const std::vector<std::vector<int>> MIN_SIZE = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    const std::vector<int> STEPS = {8, 16, 32, 64};

    int w = inputSize.width;
    int h = inputSize.height;

    std::vector<int>  feature_map_2th = {static_cast<int>(static_cast<int>((h + 1) / 2.0f) / 2.0f),
                                         static_cast<int>(static_cast<int>((w + 1) / 2.0f) / 2.0f)};
    std::vector<int>  feature_map_3th = {static_cast<int> (feature_map_2th[0] / 2.0f),
                                         static_cast<int>  (feature_map_2th[1] / 2.0f)};
    std::vector<int>  feature_map_4th = {static_cast<int> (feature_map_3th[0] / 2.0f),
                                         static_cast<int> (feature_map_3th[1] / 2.0f)};
    std::vector<int>  feature_map_5th = {static_cast<int> (feature_map_4th[0] / 2.0f),
                                         static_cast<int> (feature_map_4th[1] / 2.0f)};
    std::vector<int>  feature_map_6th = {static_cast<int> (feature_map_5th[0] / 2.0f),
                                         static_cast<int> (feature_map_5th[1] / 2.0f)};
    std::vector<std::vector<int>> feature_maps = {feature_map_3th, feature_map_4th, 
                                                  feature_map_5th, feature_map_6th};
    
    for (int k = 0; k < feature_maps.size(); k++){
        std::vector<int> feature_map = feature_maps[k];
        std::vector<int> min_sizes = MIN_SIZE[k];
        for (int i = 0; i < feature_map[0]; ++i) {
            for (int j = 0; j < feature_map[1]; ++j) {
                for (int min_size : min_sizes) {
                    float s_kx = static_cast<float>(min_size) / static_cast<float>(w);
                    float s_ky = static_cast<float>(min_size) / static_cast<float>(h);
                    float cx = (static_cast<float>(j) + 0.5) * static_cast<float>(STEPS[k]) / static_cast<float>(w);
                    float cy = (static_cast<float>(i) + 0.5) * static_cast<float>(STEPS[k]) / static_cast<float>(h);
                    priors.push_back({cx,cy,s_kx,s_ky});
                }
            }
        }
    }
}