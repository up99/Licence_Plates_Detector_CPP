#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


/// @brief function generate priors
/// @param priors:std::vector<std::vector<float>> - priors for nms 
/// @param inputSize - size of input image
/// @return void
void priorGen(std::vector<std::vector<float>> &priors, cv::Size inputSize);
