#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Namespace for filesystem operations
namespace fs = std::filesystem;

// Function to get all PNG image paths from a directory
std::vector<std::string> get_image_paths(const std::string& dir_path) {
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }
    return image_paths;
}

// Function to load the ONNX model
cv::dnn::Net load_model(const std::string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    // Set preferable backend and target for better performance
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
}

// Function to preprocess the image for the model
cv::Mat preprocess_image(const cv::Mat& img, cv::Size input_size = {240, 320 }) {
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, input_size, cv::Scalar(), true, false);
    return blob;
}

std::vector<cv::Rect> run_inference(cv::dnn::Net& net, const cv::Mat& blob) {
    std::vector<cv::Mat> outputs;
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    const cv::Mat& out = outputs[0];
    std::vector<cv::Rect> detections;
    float confidence_threshold = 0.5f;

    int num_detections = out.cols;

    // Loop over possible detections
    for (int i = 0; i < num_detections / 4; ++i) {
        float confidence = out.at<float>(0, i);

        if (confidence > confidence_threshold) {
            int base_index = i * 4;

            // Ensure we don't go beyond available data
            if (base_index + 3 >= num_detections) {
                std::cerr << "Incomplete detection at index " << i << ". Skipping.\n";
                continue;
            }

            float x_min = out.at<float>(1, base_index) * blob.cols;
            float y_min = out.at<float>(1, base_index + 1) * blob.rows;
            float x_max = out.at<float>(1, base_index + 2) * blob.cols;
            float y_max = out.at<float>(1, base_index + 3) * blob.rows;

            // Clamp to image boundaries
            x_min = std::max(0.0f, x_min);
            y_min = std::max(0.0f, y_min);
            x_max = std::min(static_cast<float>(blob.cols), x_max);
            y_max = std::min(static_cast<float>(blob.rows), y_max);

            detections.emplace_back(cv::Point(x_min, y_min),
                cv::Point(x_max, y_max));
        }
    }

    return detections;
}


void draw_detections(cv::Mat& img, const std::vector<cv::Rect>& detections, const cv::Size& input_size = { 320, 320 }) {
    double scale_x = static_cast<double>(img.cols) / input_size.width;
    double scale_y = static_cast<double>(img.rows) / input_size.height;
    for (const auto& rect : detections) {
        cv::Rect scaled_rect(
            static_cast<int>(rect.x * scale_x),
            static_cast<int>(rect.y * scale_y),
            static_cast<int>(rect.width * scale_x),
            static_cast<int>(rect.height * scale_y)
        );
        std::cout << "rect.x: " << rect.x << std::endl;
        cv::rectangle(img, scaled_rect, cv::Scalar(0, 255, 0), 2); // Green rectangle
    }
}

// Function to save the resulting image
void save_result(const cv::Mat& img, const std::string& output_dir, const std::string& filename) {
    std::string out_path = output_dir + "/" + filename;
    cv::imwrite(out_path, img);
}

// Main function
int main(int argc, char* argv[]) {
    // Check if arguments are provided correctly
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir> <model_path>" << std::endl;
        return -1;
    }

    // Get command-line arguments
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    std::string model_path = argv[3];

    // Ensure output directory exists
    fs::create_directories(output_dir);

    // Get all PNG image paths from the input directory
    auto image_paths = get_image_paths(input_dir);

    // Load the ONNX model
    auto net = load_model(model_path);
    // Process each image
    for (const auto& path : image_paths) {
        // Read the image
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }

        // Preprocess the image
        auto blob = preprocess_image(img);

        // Run inference and get detections
        auto detections = run_inference(net, blob);
        //std::cout << "Number of detections: " << detections.size() << std::endl;

        // draw bounding boxes on the image
        draw_detections(img, detections, {240, 320});

        //// save the result
        std::string filename = fs::path(path).filename().string();
        save_result(img, output_dir, filename);

        std::cout << "Processed: " << path << std::endl;
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}


