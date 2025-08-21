#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "config_model.h"

/// @brief function inference `.onnx` model
/// @param cv::dnn::Net &net - object of NN
/// @param outputs:std::vector<cv::Mat> - vector of output layers iou, conf, loc
/// @param blob:cv::Mat - blob of input image
/// @param inputSize:cv::Size - size of input image
/// @return flag:bool - true if inference is success
bool inference(cv::dnn::Net &net, std::vector<cv::Mat> &outputs, cv::Mat &blob, ModelParameters& parameters);






