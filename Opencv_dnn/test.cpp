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

int main()
{
	VideoCapture vid_capture(0);
	if (!vid_capture.isOpened())
	{
		cout << "Error opening video stream or file" << endl;
	}

	else
    {
		int fps = vid_capture.get(5);
		cout << "Frames per second :" << fps;
		int frame_count = vid_capture.get(7);
		cout << "  Frame count :" << frame_count;
	}
	while (vid_capture.isOpened())
	{
		Mat frame;
		bool isSuccess = vid_capture.read(frame);
        resize(frame, frame, Size(), 0.5, 0.5);
		if(isSuccess == true)
		{

			imshow("Frame", frame);
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