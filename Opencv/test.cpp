#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/all_layers.hpp>

// Namespace to nullify use of cv::function(); syntax
using namespace std;
using namespace cv;
string trackbarValue = "lowThreshold";
int lowThreshold = 0;
int main()
{
	VideoCapture vid_capture("aespa.mp4");
	int fps = vid_capture.get(5);
	cout << "Frames per second :" << fps;
	int frame_count = vid_capture.get(7);
	cout << "  Frame count :" << frame_count;
    Mat imgGray, imgBlur, frame, imgCan;
    
	while (vid_capture.isOpened())
	{
		bool isSuccess = vid_capture.read(frame);
        resize(frame, frame, Size(), 0.5, 0.5);
        cvtColor(frame, imgGray, COLOR_BGR2GRAY);
        GaussianBlur(frame, imgBlur, Size(7,7), 7);
        namedWindow("Edge detection", WINDOW_AUTOSIZE);
        createTrackbar("lowThreshold", "Edge detection", &lowThreshold, 255);
        Canny(imgBlur, imgCan, lowThreshold, 0, 5);
		if(isSuccess == true)
		{

			//imshow("Frame", frame);
            //imshow("Gray", imgGray);
            //imshow("Blur", imgBlur);
            imshow("Edge detection", imgCan);
		}
		if (isSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			break;
		}
		int key = waitKey(1);
		if (key == 'q')
		{
			cout << "q key is pressed by the user. Stopping the video" << endl;
			break;
		}


	}
	vid_capture.release();
	destroyAllWindows();
	return 0;
}
