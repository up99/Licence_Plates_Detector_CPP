#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

#include "config_model.h"
#include <inference.h>
#include <genpriors.h>
#include <postprocessing.h>
#include <visualization.h>


namespace fs = std::filesystem;


std::vector<std::string> get_image_paths(const std::string& dir_path) {
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }
    return image_paths;
}


void save_result(const cv::Mat& img, const std::string& output_dir, const std::string& filename) {
    std::string out_path = output_dir + "/" + filename;
    cv::imwrite(out_path, img);
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir> <model_path>" << std::endl;
        std::cerr << "EXAMPLE (run from build folder): "<< std::endl;
        std::cerr << "./main ../lp_data result ../models/model.onnx" << std::endl;
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    
    ModelParameters parameters;

    parameters.modelPath = argv[3] ;

    fs::create_directories(output_dir);

    auto image_paths = get_image_paths(input_dir);

    for (const auto& image_path : image_paths) {

        cv::Mat img = cv::imread(image_path.c_str(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: Could not read image." << std::endl;
            return -1;
        }
        
        cv::Mat originalImg = img.clone();
        cv::Size originalSize(img.cols, img.rows);

        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0, parameters.inputSize, cv::Scalar(0,0,0), true, false);

        cv::dnn::Net net;
        std::vector<cv::Mat> outputs;


        bool flag_inf = inference(net, outputs, blob, parameters);

        if (!flag_inf) {
            std::cerr << "Error: Inference failed." << std::endl;
            return -1;
        }

        std::vector<std::vector<float>> priors;

        priorGen(priors, parameters.inputSize);

        std::vector<std::vector<float>> finalDetections;

        postprocessing(outputs,priors,finalDetections,parameters);


        float scale_x = static_cast<float>(originalSize.width) / parameters.inputW;
        float scale_y = static_cast<float>(originalSize.height) / parameters.inputH;

        float scale_xy[2] = {scale_x, scale_y};

        visualization(originalImg, finalDetections, scale_xy);
        

        cv::imshow("Detections", originalImg);
        
        std::string filename = fs::path(image_path).filename().string();
        save_result(originalImg, output_dir, filename);

        cv::waitKey(0);
        
    }

        

    return 0;
}