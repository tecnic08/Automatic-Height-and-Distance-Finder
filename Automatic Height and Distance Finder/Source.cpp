#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

bool pointTrackingFlag = false;
bool calculateTrackpointFlag = false;
bool clearTrackingFlag = false;
Point2f currentPoint;
vector<Point2f> desiredPoint;

// Good Point border criteria default is 160, 480, 240
int borderLeft = 220, borderRight = 420, borderLower = 240, borderHigh = 10;

// Detect mouse events
void onMouse(int event, int x, int y, int, void*)
{
	// Show mouse position
	/*if (event == CV_EVENT_MOUSEMOVE)
	{
	cout << "(" << x << "," << y << ")" << endl;
	}*/
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		currentPoint = Point2f((float)x, (float)y);
		pointTrackingFlag = true;
	}
	if (event == CV_EVENT_RBUTTONDOWN)
	{
		calculateTrackpointFlag = true;
	}
	if (event == CV_EVENT_MBUTTONDOWN)
	{
		clearTrackingFlag = true;
	}
}

int main(int argc, char* argv[])
{
	// Open camera
	VideoCapture cap(0);
	// Check if the camera is open yet
	if (!cap.isOpened())
	{
		cerr << "Unable to open the webcam." << endl;
		return -1;
	}

	// Height and distace calculation variable
	float xDist, height, cameraHeight, deltaX;
	deltaX = 0.5;
	cameraHeight = 0.32;

	// Set the desired point grid 
	int desiredX[6] = {160,224,288,352,416,480};
	int desiredY[5] = {60,100,140,180,220};

	// Push desired (x,y) in vector of desiredPoint
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			desiredPoint.push_back(Point2f(desiredX[i], desiredY[j]));
		}
	}

	TermCriteria terminationCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.02);

	// Matching box size
	Size windowSize(25, 25);

	// Max number of points
	const int maxNumPoints = 200;

	string windowName = "Height and Range finder";
	namedWindow(windowName, 1);
	setMouseCallback(windowName, onMouse, 0);

	Mat prevGrayImage, curGrayImage, image, frame;
	vector<Point2f> trackingPoints[2];
	vector<Point2f> calculatePoints[2];
	vector<int> goodPointsVecTransfer;

	// Image size scaling factor
	float scalingFactor = 1.0;

	while (true)
	{
		cap >> frame;

		if (frame.empty())
			break;

		resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

		frame.copyTo(image);

		cvtColor(image, curGrayImage, COLOR_BGR2GRAY);

		if (!trackingPoints[0].empty())
		{
			vector<uchar> statusVector;
			vector<float> errorVector;

			if (prevGrayImage.empty())
			{
				curGrayImage.copyTo(prevGrayImage);
			}

			calcOpticalFlowPyrLK(prevGrayImage, curGrayImage, trackingPoints[0], trackingPoints[1], statusVector, errorVector, windowSize, 3, terminationCriteria, 0, 0.001);

			int count = 0;
			int minDist = 7;
			int goodPoints = 0;
			vector<int> goodPointsVec;
			// For showing tracking point number
			stringstream bufferstring;
			string gg;

			for (int i = 0; i < trackingPoints[1].size(); i++)
			{
				if (pointTrackingFlag)
				{	// Check if new point are too close.
					if (norm(currentPoint - trackingPoints[1][i]) <= minDist)
					{
						pointTrackingFlag = false;
						continue;
					}
				}

				// Check if the status vector is good
				if (!statusVector[i])
					continue;
				// Remove tracking point that is out of ROI
				if (trackingPoints[1][i].x < borderLeft || trackingPoints[1][i].x > borderRight)
					continue;
				if (trackingPoints[1][i].y < borderHigh || trackingPoints[1][i].y > borderLower)
					continue;

				// Point optimization (removed)
				//trackingPoints[1][count++] = trackingPoints[1][i];

				// Track point icon
				int radius = 8;
				int thickness = 2;
				int lineType = 3;
				circle(image, trackingPoints[1][i], radius, Scalar(0, 255, 0), thickness, lineType);
				bufferstring.str("");
				bufferstring << i;
				gg = bufferstring.str();
				cv::putText(image, gg, trackingPoints[1][i], CV_FONT_NORMAL, 0.6, Scalar(0, 255, 0), 1, 1);
				goodPoints++;
				goodPointsVec.push_back(i);

			}
			// Point optimization (removed)
			//trackingPoints[1].resize(count);

			// Reset grid if there are too little point
			/*if (goodPoints <= 4)
			{
				clearTrackingFlag = true;
				pointTrackingFlag = true;
				cout << "Tracking grid reset" << endl;
			}*/

			goodPointsVecTransfer = goodPointsVec;
		}

		// Calculate the distance
		if (calculateTrackpointFlag)
		{
			for (int i = 0; i < goodPointsVecTransfer.size(); i++)
			{
				xDist = (tan(-0.00305*calculatePoints[0][goodPointsVecTransfer[i]].y + 0.678)*deltaX)
					/ (tan(-0.00305*trackingPoints[1][goodPointsVecTransfer[i]].y + 0.678)
						- tan(-0.00305*calculatePoints[0][goodPointsVecTransfer[i]].y + 0.678));
				height = xDist*tan(-0.00305*trackingPoints[1][goodPointsVecTransfer[i]].y + 0.678) + cameraHeight;

				if (xDist >= 15)
					cout << "Point " << goodPointsVecTransfer[i] << " too far." << endl;
				else if (xDist < 0)
					cout << "Point " << goodPointsVecTransfer[i] << " cannot be calculated." << endl;
				else
					cout << "Point " << goodPointsVecTransfer[i] << ", height is " << height << "m and it is " << xDist << "m away." << endl;
			}
			cout << endl;
			calculateTrackpointFlag = false;
		}

		// Reset the tracking point
		if (clearTrackingFlag)
		{
			trackingPoints[0].clear();
			trackingPoints[1].clear();
			calculatePoints[0].clear();
			calculatePoints[1].clear();
			goodPointsVecTransfer.clear();

			clearTrackingFlag = false;
		}

		// Refining the location of the feature points
		if (pointTrackingFlag && trackingPoints[1].size() < maxNumPoints)
		{
			for (int k = 0; k < desiredPoint.size(); k++)
			{
				vector<Point2f> tempPoints;
				tempPoints.push_back(desiredPoint[k]);

				cornerSubPix(curGrayImage, tempPoints, windowSize, cvSize(-1, -1), terminationCriteria);
				
				// Add point for calculation
				calculatePoints[0].push_back(tempPoints[0]);
				trackingPoints[1].push_back(tempPoints[0]);
			}

			pointTrackingFlag = false;
		}

		imshow(windowName, image);

		// ESC Check
		char ch = waitKey(10);
		if (ch == 27)
			break;

		// Update 'previous' to 'current' point vector
		std::swap(trackingPoints[1], trackingPoints[0]);

		// Update previous image to current image
		cv::swap(prevGrayImage, curGrayImage);
	}

	return 0;
}

