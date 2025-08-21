#include "visualization.h"



void visualization(cv::Mat &originalImg, 
    std::vector<std::vector<float>> &finalDetections,
    float* scale_xy){

        for (auto det : finalDetections) {

            std::vector<cv::Point> points;
    
            for (int i = 0; i < 4; ++i) {
                points.emplace_back(
                static_cast<int>(det[i * 2] * scale_xy[0]),
                static_cast<int>(det[i * 2 + 1] * scale_xy[1])
                );
            }
    
            for (const auto& p : points) {
                cv::circle(originalImg, p, 5, cv::Scalar(0, 255, 0), -1);
            }

            for (size_t i = 0; i < points.size(); ++i) {
                cv::line(originalImg, points[i], points[(i + 1) % points.size()], cv::Scalar(0, 0, 255), 2);
            }

            std::string label = cv::format("%.2f", det[8]);
            cv::putText(originalImg, label, points[0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        }
}