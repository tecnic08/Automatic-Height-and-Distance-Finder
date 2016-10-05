/*Copyright Noppawit Lertutsahakul
	This program will find height and distance of a defined point. Accuracy depends on the height of the object and curvature of the lens.

	+++ Memo of things to do +++
	- Make this code runs on Ubuntu
	- Make ROS node that publish odometry data and connnect it with this code.
	- Make this code autonomous
		- Automatically deploy a grid
		- If the point has been moved for more than some pixel (ex. 50)
		  then return it to the original point (Not sure if this is a good idea)
		- Or if the point moved for more than some pixel, add a new point to that black point.
		- Other idea
		- Vector of former tracking point must also contain location data not only point.
		- After sometime, tracking point vector will be very large, it needs to be cleared.
	- Test this code on the onboard computer of Kenaf

*/
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

// Good Point border criteria
int borderLeft = 120, borderRight = 520, borderLower = 210, borderUpper = 10;

// Camera y pixel vs angle slope equation (Linear Equation) refer to excel file
// [angle = Ay + B]
float aConstant = -0.00305;
float bConstant = 0.678;

// deltaX is how far the camera has moved, cameraHeight is the hight of the camera from the ground
float deltaX = 0.5;
float cameraHeight = 0.32;
float xDist, xDistC, height;

// Distance Error Correction (Parabolic Equation) refer to excel file
// [Error Percentage = cConstant y^2 + dConstant y +eConstant]
// Turn this function on or off with the errorCompensation bool variable
float cConstant = 0.0025;
float dConstant = -0.6445;
float eConstant = 45.775;
bool errorCompensation = true;

// Set the desired point grid
// For 640x480
int desiredX[9] = { 160,200,240,280,320,360,400,440,480 };
int desiredY[5] = { 60,100,140,180,200 };


// Declaring some flags
bool pointTrackingFlag = false;
bool calculateTrackpointFlag = false;
bool clearTrackingFlag = false;
bool recenterOffGridPoint = false;

Point2f currentPoint;
vector<Point2f> desiredPoint;

int pointNeedsRecenter;

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
	VideoCapture cap(1);

	// Check whether the camera is open yet
	if (!cap.isOpened())
	{
		cerr << "Unable to open the webcam." << endl;
		return -1;
	}

	// Push desired (x,y) in vector of desiredPoint
	for (int i = 0; i < 9; i++)
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
	// trackingPoints is the current point.
	vector<Point2f> trackingPoints[2];
	// calculatePoints is the previous point data that will be used for calculation
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

				// Check if the status vector is good if not, skip the code below
				if (!statusVector[i])
				{
                    recenterOffGridPoint = true;
                    pointNeedsRecenter = i;
					continue;
                }
				// Remove tracking point that is out of ROI
				if (trackingPoints[1][i].x < borderLeft || trackingPoints[1][i].x > borderRight)
				{
                    recenterOffGridPoint = true;
                    pointNeedsRecenter = i;
					continue;
                }
				if (trackingPoints[1][i].y < borderUpper || trackingPoints[1][i].y > borderLower)
                {
                    recenterOffGridPoint = true;
                    pointNeedsRecenter = i;
					continue;
                }

				// Point optimization (removed)
				//trackingPoints[1][count++] = trackingPoints[1][i];

				// Track point icon
				int radius = 8;
				int thickness = 2;
				int lineType = 3;
				circle(image, trackingPoints[1][i], radius, Scalar(0, 255, 0), thickness, lineType);

				// Show point number in frame
				bufferstring.str("");
				bufferstring << i;
				gg = bufferstring.str();
				cv::putText(image, gg, Point(trackingPoints[1][i].x + 5,trackingPoints[1][i].y + 5), CV_FONT_NORMAL, 0.6, Scalar(0, 255, 0), 1, 1);

				// Add goodPoints count and save the point index in goodPointsVec for calculation
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

			// Transfer local vector variable to global vector variable
			goodPointsVecTransfer = goodPointsVec;
		}

		// Calculate the distance
		if (calculateTrackpointFlag)
		{
			// Set float point decimal point
			std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
			std::cout.precision(2);

			for (int i = 0; i < goodPointsVecTransfer.size(); i++)
			{
				// xDist calculation (How far is it from the object)
				xDist = (tan(aConstant*calculatePoints[0][goodPointsVecTransfer[i]].y + bConstant)*deltaX)
					/ (tan(aConstant*trackingPoints[1][goodPointsVecTransfer[i]].y + bConstant)
						- tan(aConstant*calculatePoints[0][goodPointsVecTransfer[i]].y + bConstant));

				// height calculation (How high is the object)
				height = xDist*tan(aConstant*trackingPoints[1][goodPointsVecTransfer[i]].y + bConstant) + cameraHeight;

				if (errorCompensation)
				{
					// xDist error compensation
					xDistC = xDist - abs((xDist*(((cConstant*calculatePoints[0][goodPointsVecTransfer[i]].y*calculatePoints[0][goodPointsVecTransfer[i]].y)
						+ (dConstant*calculatePoints[0][goodPointsVecTransfer[i]].y) + eConstant) / 100)));
				}

				if (xDist < 0 || xDist >= 15)
					cout << "Point " << goodPointsVecTransfer[i] << "(" << calculatePoints[0][goodPointsVecTransfer[i]].x << ","
					<< calculatePoints[0][goodPointsVecTransfer[i]].y << ") "<< " cannot be calculated." << endl;
				else
				{
					if (errorCompensation)
					{
						cout << "Point " << goodPointsVecTransfer[i] << "(" << calculatePoints[0][goodPointsVecTransfer[i]].x << ","
							<< calculatePoints[0][goodPointsVecTransfer[i]].y << ") height is " << height << "m and it is " << xDistC << "m (" << xDist << "m ) away." << endl;
					}
					else
					{
						cout << "Point " << goodPointsVecTransfer[i] << "(" << calculatePoints[0][goodPointsVecTransfer[i]].x << ","
							<< calculatePoints[0][goodPointsVecTransfer[i]].y << ") height is " << height << "m and it is " << xDist << "m  away." << endl;
					}
				}
			}
			// Add blank line tos separate each iteration
			cout << endl;

			calculateTrackpointFlag = false;
		}

		// Reset the tracking point
		if (clearTrackingFlag)
		{
            // Turn off recentering otherwise segmentation fault will occur
            recenterOffGridPoint = false;

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

				// Add point for calculation.
				calculatePoints[0].push_back(tempPoints[0]);
				trackingPoints[1].push_back(tempPoints[0]);
			}

			pointTrackingFlag = false;
		}

		// Tracking point is bad or moved away from border, reset that point.
		if (recenterOffGridPoint)
		{
            vector<Point2f> tempPoints;
            tempPoints.push_back(desiredPoint[pointNeedsRecenter]);

            cornerSubPix(curGrayImage, tempPoints, windowSize, cvSize(-1, -1), terminationCriteria);

            // Remove old, bad tracking point from the vector.
            calculatePoints[0].erase(calculatePoints[0].begin()+pointNeedsRecenter);
            trackingPoints[1].erase(trackingPoints[1].begin()+pointNeedsRecenter);

            // Insert new tracking point into the vector.
            calculatePoints[0].insert(calculatePoints[0].begin()+pointNeedsRecenter, tempPoints[0]);
            trackingPoints[1].insert(trackingPoints[1].begin()+pointNeedsRecenter, tempPoints[0]);

            cout << "Point recentered " << pointNeedsRecenter << endl;

            recenterOffGridPoint = false;
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

