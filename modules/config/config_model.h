#pragma once

#include <string>
#include <opencv2/opencv.hpp>

/// @brief Model parameters struct which consist:
/// @param modelPath:std::string - path to model.onnx
/// @param conf_threshold:float - confidence threshold
/// @param nms_threshold:float - NMS threshold
/// @param input_w:int - input width
/// @param input_h:int - input height
/// @param inputSize:cv::Size - input size of picture
struct ModelParameters
{
    std::string modelPath = "../models/model.onnx";
    const float confThreshold           = 0.3;
    const float nmsThreshold            = 0.5;
    const int  inputW                   = 320;
    const int  inputH                   = 240;
    const cv::Size inputSize            = cv::Size(inputW, inputH);
};

