#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/// @brief This function visualizes the detections on the original image.
/// @param originalImg:cv::Mat
/// @param finalDetections:std::vector<std::vector<float>>
/// @param scale_xy:float[2] massive of scale factors for x and y
void visualization(cv::Mat &originalImg, 
                   std::vector<std::vector<float>> &finalDetections,
                   float* scale_xy);