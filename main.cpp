#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

void postprocessClassification(const cv::Mat& output);

int main()
{
    auto modelPath = "/Users/boris.pouw/Documents/better.onnx";

    // Preprocess and load your potato image
    cv::Mat image = cv::imread("../images/blauw.jpg");

    // Resize the image to the network input size
    cv::Size netInputSize(128, 128);
    cv::resize(image, image, netInputSize);

    // Normalize the image
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, netInputSize, cv::Scalar(0, 0, 0), true, false);

    // Set up net
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // Input blob in net
    net.setInput(blob);

    // Forward
    const cv::Mat &output = net.forward();

    // Process classification
    postprocessClassification(output);
}


void postprocessClassification(const cv::Mat& output) {
    // Convert the output cv::Mat to a vector
    std::vector<float> output_values;
    output_values.assign(output.begin<float>(), output.end<float>());

    // Find the index with the highest probability (argmax)
    auto max_index = std::distance(output_values.begin(), std::max_element(output_values.begin(), output_values.end()));

    // Print the predicted class and its probability
    std::cout << "Predicted Class: " << max_index << std::endl;
    std::cout << "Probability: " << output_values[max_index] << std::endl;

    const std::vector<std::string> class_labels = {"Blauw", "Dierlijk", "Gezond", "Groen", "Hol", "Inwendig bruin", "Rooibeschadiging", "Rot"};

    // Print the predicted class label
    std::cout << "Predicted Class Label: " << class_labels[max_index] << std::endl;
}