#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <config_model.h>


/// @brief decode the output from network
/// @param loc 
/// @param conf -confidance vector
/// @param iou  - iou vector    
/// @param priors - priors vector
/// @param inputSize - parameter of input size
/// @return void
void decode (const std::vector<std::vector<float>> &loc,
    const std::vector<std::vector<float>> &conf,
    std::vector<std::vector<float>> &iou, 
    std::vector<std::vector<float>> &dets,
    const std::vector<std::vector<float>> &priors,
    const cv::Size& inputSize);

/// @brief decode `std::vector<cv::Mat>` to `std::vector<std::vector<float>>`
/// @param mat:cv::Mat
/// @param vec:vector<float>
/// @return bool : true if success, false if failed
bool matToVector(const cv::Mat& mat, std::vector<std::vector<float>>& vec);


/// @brief main code of postprocessing
/// @param outputs:std::vector<cv::Mat> - output from network
/// @param priors:std::vector<std::vector<float>> - priors generated from anchors
/// @param final_detection:std::vector<std::vector<float>> - final detection result
/// @param parameters:struct - parameters of model 
void postprocessing(std::vector<cv::Mat> &outputs, 
                    std::vector<std::vector<float>> &priors,
                    std::vector<std::vector<float>> &final_detection,  
                    ModelParameters &parameters);

