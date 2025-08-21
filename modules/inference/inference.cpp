#include "inference.h"



bool inference(cv::dnn::Net &net, std::vector<cv::Mat> &outputs, cv::Mat &blob, ModelParameters& parameters){
    
    try {
        net = cv::dnn::readNet(parameters.modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "Error: Could not load model: " << e.what() << std::endl;
        return false;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    return true;
}