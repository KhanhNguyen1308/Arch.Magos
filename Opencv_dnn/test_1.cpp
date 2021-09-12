#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 2;

// colors for bounding boxes
const Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);
int main()
{
    vector<string> class_names;
    {
        ifstream class_file("classes.txt");
        if (!class_file)
        {
            cerr << "failed to open classes.txt\n";
            return 0;
        }

       string line;
        while (getline(class_file, line))
            class_names.push_back(line);
    }

    auto net = readNetFromDarknet("cfg/yolov4-tiny-448-2.cfg", "model/yolov4-tiny-448-2.weights");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();
    bool re = false;
    Mat frame=imread("img48.jpg"); 
    Mat blob;
    vector<Mat> detections;
    auto total_start = chrono::steady_clock::now();
    blobFromImage(frame, blob, 0.00392, Size(448, 448), Scalar(), true, false, CV_32F);
    net.setInput(blob);
    auto dnn_start = chrono::steady_clock::now();
    net.forward(detections, output_names);
    auto dnn_end = chrono::steady_clock::now();
    vector<int> indices[NUM_CLASSES];
    vector<Rect> boxes[NUM_CLASSES];
    vector<float> scores[NUM_CLASSES];

    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            Rect rect(x - width/2, y - height/2, width, height);
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++)
    {
        NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
    }
    for (int c= 0; c < NUM_CLASSES; c++)
    {
        for (size_t i = 0; i < indices[c].size(); ++i)
        {
            const auto color = colors[c % NUM_COLORS];
            auto idx = indices[c][i];
            const auto& rect = boxes[c][idx];
            rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);
            ostringstream label_ss;
            label_ss << class_names[c] << ": " << fixed << setprecision(2) << scores[c][idx];
            auto label = label_ss.str();
            int baseline;
            auto label_bg_sz = getTextSize(label.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);
            putText(frame, label.c_str(), Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
        }
    }
    auto total_end = chrono::steady_clock::now();
    float inference_fps = 1000.0 / chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
    float total_fps = 1000.0 / chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    ostringstream stats_ss;
    stats_ss << fixed << std::setprecision(2);
    stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    auto stats = stats_ss.str();
    int baseline;
    auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    rectangle(frame, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
    putText(frame, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));
    imshow("output", frame);
    int key=waitKey(0);
    return 0;
}
