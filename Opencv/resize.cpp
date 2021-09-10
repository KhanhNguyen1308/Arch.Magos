// Import Packages
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
int maxScaleUp = 100;
int scaleFactor = 1;
string windowName = "Resize Image";
string trackbarValue = "Scale";
void scaleImage(int, void*)
{
	Mat image = imread("../../Input/sample.jpg");
    double scaleFactorDouble = 1 + scaleFactor/100.0;
    if (scaleFactorDouble == 0)
	{
        scaleFactorDouble = 1;
    }
    Mat scaledImage;
    resize(image, scaledImage, Size(), scaleFactorDouble, scaleFactorDouble, INTER_LINEAR);
    imshow(windowName, scaledImage);
}

int main()
{
Mat image = imread("../../Input/sample.jpg");
namedWindow(windowName, WINDOW_AUTOSIZE);
createTrackbar(trackbarValue, windowName, &scaleFactor, maxScaleUp, scaleImage);
scaleImage(25,0);
imshow(windowName, image);
waitKey(0);
destroyAllWindows();
return 0;
}