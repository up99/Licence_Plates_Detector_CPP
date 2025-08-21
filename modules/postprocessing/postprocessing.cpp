#include "postprocessing.h"

void decode(const std::vector<std::vector<float>> &loc,
      const std::vector<std::vector<float>> &conf,
      std::vector<std::vector<float>> &iou, 
      std::vector<std::vector<float>> &dets,
      const std::vector<std::vector<float>> &priors,
      const cv::Size& inputSize){
        
        dets.reserve(priors.size());
        float variance[2] = {0.1, 0.2}; 

        std::vector<float> scores;
        scores.reserve(priors.size());

        for (size_t i = 0; i < iou.size(); ++i) {
            float iou_val = iou[i][0];

            if (iou_val < 0) {
                iou_val = 0.0f;
            }
            if (iou_val > 1){
                iou_val = 1.0f;
            }
        scores.push_back(std::sqrt(conf[i][1] * iou_val));
        }

        for (size_t i = 0; i < priors.size(); ++i) {
            std::vector<float> det(9);

            float cx = priors[i][0];
            float cy = priors[i][1];
            float s_kx = priors[i][2];
            float s_ky = priors[i][3];

            det[0] = (cx + loc[i][4]  * variance[0] * s_kx) * inputSize.width;
            det[1] = (cy + loc[i][5]  * variance[0] * s_ky) * inputSize.height;
            det[2] = (cx + loc[i][6]  * variance[0] * s_kx) * inputSize.width;
            det[3] = (cy + loc[i][7]  * variance[0] * s_ky) * inputSize.height;
            det[4] = (cx + loc[i][10] * variance[0] * s_kx) * inputSize.width;
            det[5] = (cy + loc[i][11] * variance[0] * s_ky) * inputSize.height;
            det[6] = (cx + loc[i][12] * variance[0] * s_kx) * inputSize.width;
            det[7] = (cy + loc[i][13] * variance[0] * s_ky) * inputSize.height;
            det[8] = scores[i];

            dets.push_back(det);
        }
}


bool matToVector(const cv::Mat& mat, std::vector<std::vector<float>>& vec) {
    
    if (mat.dims != 2) {
        std::cerr << "Error: matToVector expects a 2D matrix." << std::endl;
        return false;
    }

    int rows = mat.size[0];
    int cols = mat.size[1];
    vec.clear();
    vec.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        const float* row_ptr = mat.ptr<float>(i);
        vec.emplace_back(row_ptr, row_ptr + cols);
    }
    return true;
}


void postprocessing(std::vector<cv::Mat> &outputs, 
                    std::vector<std::vector<float>> &priors,
                    std::vector<std::vector<float>> &final_detections,  
                    ModelParameters &parameters){

    cv::Mat loc_mat  = outputs[2];
    cv::Mat conf_mat = outputs[0];
    cv::Mat iou_mat  = outputs[1];

    std::vector<std::vector<float>> loc_vec, conf_vec, iou_vec;

    bool flag_loc  = matToVector(loc_mat, loc_vec);
    bool flag_conf = matToVector(conf_mat, conf_vec);
    bool flag_iou  = matToVector(iou_mat, iou_vec);

    

    std::vector<std::vector<float>> dets;

    if (flag_loc && flag_conf && flag_iou){
        decode(loc_vec, conf_vec, iou_vec, dets, priors, parameters.inputSize);
    }



    std::vector<cv::Rect> boxes_for_nms;
    std::vector<float> scores_for_nms;
    std::vector<std::vector<float>> filtered_dets;

    
    for (const auto& det : dets) {
        float score = det[8];
        if (score > parameters.confThreshold) {
            float min_x = std::min({det[0], det[2], det[4], det[6]});
            float max_x = std::max({det[0], det[2], det[4], det[6]});
            float min_y = std::min({det[1], det[3], det[5], det[7]});
            float max_y = std::max({det[1], det[3], det[5], det[7]});
            
            boxes_for_nms.emplace_back(
                static_cast<int>(min_x),
                static_cast<int>(min_y),
                static_cast<int>(max_x - min_x),
                static_cast<int>(max_y - min_y)
            );
            scores_for_nms.push_back(score);
            filtered_dets.push_back(det);
        }
    }
    std::vector<int> final_indices;
    if (!boxes_for_nms.empty()) {
        cv::dnn::NMSBoxes(boxes_for_nms, scores_for_nms, parameters.confThreshold, parameters.nmsThreshold, final_indices);
    }

    for (int idx : final_indices){
        final_detections.push_back(filtered_dets[idx]);
    }
}
