
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "lowLevelFunc.h"
#include "highLevelFunc.h"

using namespace cv;
using namespace std;


//Food Selector uses Canny edges, gray pixels removal and some dilation.
//Food has a lot of tiny edges in it, so I take the canny edges and dilate them to get a good mask.
//Then I remove the gray pixels (that usually is background) and re-dilate to have smoother result.
//A partial result is saved because bread detection works better on it
Mat foodSelector(Mat image, Mat& partialResult){
	
	//create needed Mat objects
	Mat gray = image.clone();
	cvtColor(gray, gray, COLOR_BGR2GRAY);
	Mat finalImage = Mat::zeros(image.size(), CV_8UC3);
	
	//use canny to find borders
	Canny(gray, gray, 100, 50);
	
	//dilate the borders to select the areas around them	
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(21,21)));
	
	//use dilated borders as mask on original image
	image.copyTo(finalImage, gray);

	//delete gray areas (the difference between the pixel values must be at least 58)
	removeGray(finalImage, 58);
	
	//save the partial result that is needed in bread detection
	partialResult = finalImage.clone();
	
	//re-convert in gray, re-dilate and blur to create a new smoother mask
	gray = finalImage.clone();
	cvtColor(gray, gray, COLOR_BGR2GRAY);
	
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	GaussianBlur(gray, gray, Size(7, 7), 0, 0);

	//use the new mask on the original image
	finalImage = Mat::zeros(image.size(), CV_8UC3);
	image.copyTo(finalImage, gray);
	
	return finalImage;
}


//Plates are found with hough circles function
//This is useful to isolate only real food and delete other selected pixels
vector<Vec3f> platesFinder(Mat image, Mat foodImage){

	//create image objects
	Mat hough = Mat::zeros(image.size(), CV_8UC3);
	Mat gray = Mat(image.rows, image.cols, CV_8UC1);
	
	//remove potential food pixels by creating a reversed mask on foodImage
	Mat mask = foodImage.clone();
	cvtColor(mask, mask, COLOR_BGR2GRAY);
	threshold(mask, mask, 0, 255, THRESH_BINARY);
	bitwise_not(mask, mask);
	image.copyTo(hough, mask);
	
	//calculate borders with canny
	Canny(hough, gray, 77, 289);
	
	//calculate circles with hough
	vector<Vec3f> circles, remaining;
	int minDist = 227, houghTh = 70, minR = 87, maxR = 191;
	float dp = 1.6;
	HoughCircles(gray, circles, HOUGH_GRADIENT, dp, minDist, 77, houghTh, minR, maxR);
	
	//delete empty circles
	for(Vec3f circle : circles){
		Point center(cvRound(circle[0]), cvRound(circle[1]));
		int radius = cvRound(circle[2]);
		
		if(checkValid(center, radius, foodImage) >= 0)
			remaining.push_back(circle);
	}
	
	//delete overlapping circles
	//there can't be more than 3 so delete until only 3 are left
	//delete even if there are 3 or less if there is one circle with high overlapp
	if(remaining.size() > 3)
		remaining = deleteMostOverlapping(remaining, true);
	else
		remaining = deleteMostOverlapping(remaining, false);
	
	return remaining;
}


//Other than the food in the plates the project require to recognise only the bread
//To do that this function remove all the food that is inside the plates and try to
//find pixels with color similat to the bread among the remaining ones. It outputs an 
//image with nothing outside the plates and return a point that indicates where the bread
//is, or (-1,-1) if the bread was not found
Point breadDetector(Mat image, vector<Vec3f> plates, Mat partialResult, Mat& outputImage){
	
	//remove plates content
	Mat bread = togglePlatesContent(partialResult, plates, true);
	
	//keep only bread-colored pixels	
	bread = keepBread(bread);
	
	//erode and dilate multiple times to remove most pixels but not the bread ones
	Mat gray = bread.clone();
	cvtColor(gray, gray, COLOR_BGR2GRAY);
	
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(53,53)));
	
	//use result as a mask for the original image
	bread = Mat::zeros(image.size(), CV_8UC3);
	image.copyTo(bread, gray);
	
	//sometimes it keeps not only the bread but also the borders of the tray
	//to erase them here is reused the technique of the foodSelector
	//the bread has a lot of tiny edges around it, while the tray doesn't
	
	//detect edges
	Canny(bread, gray, 162, 131);
	
	//dilate and erode
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(13,13)));	
	
	//use the dilated borders and find the median coordinate of non-zero pixels
	Point minP (image.cols, image.rows), maxP (-1, -1);
	for(int j=0; j<image.cols*image.rows; j++) {
		
		int x=j%image.cols, y=j/image.cols;
		int valGray = gray.at<uchar>(y,x);
		
		if(valGray != 0){
			if(minP.x>x)
				minP.x=x;
			if(minP.y>y)
				minP.y=y;
			if(maxP.x<x)
				maxP.x=x;
			if(maxP.y<y)
				maxP.y=y;
		}
	}
	
	//remove non-plates content
	outputImage = togglePlatesContent(outputImage, plates, false);
	
	//return median point of bread pixels (or (-1,-1) if no bread pixels are found)
	if(maxP.x != -1){
		Point median (minP.x+((maxP.x-minP.x)/2), minP.y+((maxP.y-minP.y)/2));
		return median;
	}else{
		Point nullPoint (-1, -1);
		return nullPoint;
	}
}


//Once the bread position is found, this function try to select and show all the bread pixels.
//Returns an image with food in plates and the founded bread.
//Note that the function is very long and uses a lot of different techniques combined together,
//testing showed that every section of this function contribuites to the result in a very strong
//way, so it was not possible to semplify it.
Mat breadSelector(Mat image, Mat foodImg, Point breadPos){
	
	//clone so the original image is not modified
	Mat foodImage = foodImg.clone();
	
	//search for bread for 120 pixels around the point
	int size = 120;
	
	//crop image around bread point
	Mat breadImg = Mat::zeros(min(size,breadPos.y)+min(size,foodImage.rows-breadPos.y), min(size,breadPos.x)+min(size,foodImage.cols-breadPos.x), CV_8UC3);
	Point start (max(breadPos.x-size, 0), max(0,breadPos.y-size));
	
	for(int j=0; j<breadImg.rows; j++) {
		for(int k=0; k<breadImg.cols; k++) {
			int x = start.x+k, y=start.y+j;
			if ((x - breadPos.x)*(x - breadPos.x) + (y - breadPos.y)*(y - breadPos.y) <= size*size)
				breadImg.at<Vec3b>(j, k) = image.at<Vec3b>(y, x);
		}
	}
	
	//converto to YCrCb, so bread color is more easy to spot
	Mat gray (breadImg.size(), CV_8UC1);
	Mat breadCopy = breadImg.clone();
	
	Mat channels[3];
	cvtColor(breadCopy, breadCopy, COLOR_BGR2YCrCb);
	
	//split the 3 channels
	split(breadCopy, channels);
	cvtColor(breadCopy, breadCopy, COLOR_YCrCb2BGR);
	
	//equalize and treshold channel 1 (bread color is very bright)
	equalizeHist(channels[1], channels[1]);
	inRange(channels[1], 234, 243, channels[1]);
	
	//adjust selected pixels
	erode(channels[1], channels[1], getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
	dilate(channels[1], channels[1], getStructuringElement(MORPH_ELLIPSE, Size(57,57)));
	erode(channels[1], channels[1], getStructuringElement(MORPH_ELLIPSE, Size(65,65)));
	
	//equalize and treshold channel 2 (bread color is very dark)
	equalizeHist(channels[2], channels[2]);
	inRange(channels[2], 25, 34, channels[2]);
	
	//adjust selected pixels
	erode(channels[2], channels[2], getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
	dilate(channels[2], channels[2], getStructuringElement(MORPH_ELLIPSE, Size(51,51)));
	erode(channels[2], channels[2], getStructuringElement(MORPH_ELLIPSE, Size(55,55)));
	
	//combine the two results and save in channel 0
	Point Min (breadCopy.cols, breadCopy.rows), Max (0,0);
	bitwise_or(channels[1], channels[2], channels[0]);
	
	//adjust selected pixels
	dilate(channels[0], channels[0], getStructuringElement(MORPH_ELLIPSE, Size(11,11)));
	
	//find median center
	for (int x = 0; x < breadCopy.cols; x++) {
		for (int y = 0; y < breadCopy.rows; y++) {
			uchar val = channels[0].at<uchar>(y,x);
			
			if(val>0){
				Min.x = x < Min.x? x : Min.x;
				Min.y = y < Min.y? y : Min.y;
				Max.x = x > Max.x? x : Max.x;
				Max.y = y > Max.y? y : Max.y;
			}
		}
	}
	Point median (Min.x+((Max.x-Min.x)/2), Min.y+((Max.y-Min.y)/2));
	
	//find maximum radius
	int radius=0;
	for (int x = 0; x < breadCopy.cols; x++) {
		for (int y = 0; y < breadCopy.rows; y++) {
			uchar val = channels[0].at<uchar>(y,x);
			
			if(val>0){
				Point here (x,y);
				int dist = sqrt(pow(here.x-median.x,2)+pow(here.y-median.y,2));
				radius = radius < dist? dist : radius;
			}
		}
	}
	
	//dilate a lot and then crop on maximum radius (this makes the selection more round while tring to preserve the shape)
	dilate(channels[0], channels[0], getStructuringElement(MORPH_ELLIPSE, Size(139,139)));
	
	for (int x = 0; x < breadCopy.cols; x++) {
		for (int y = 0; y < breadCopy.rows; y++) {
			if ((x - median.x)*(x - median.x) + (y - median.y)*(y - median.y) > radius*radius) {
				channels[0].at<uchar>(y,x) = 0;
			}
		}
	}
	
	//use selection as a mask
	breadCopy = Mat::zeros(foodImage.rows, foodImage.cols, CV_8UC3);
	breadImg.copyTo(breadCopy, channels[0]);
	
	//improve selection by using the same approach of the foodSelector (bread has a lot of tiny borders)
	
	//calculate borders with canny
	gray = breadCopy.clone();
	
	cvtColor(gray, gray, COLOR_BGR2GRAY);
	Canny(gray, gray, 94, 13);
	
	//adjust selection
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(9,9)));
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(13,13)));
	erode(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(11,11)));
	dilate(gray, gray, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	
	//find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	//make a mask of a countour that contains the bread
	Mat mask = Mat::zeros(breadCopy.rows, breadCopy.cols, CV_8UC1);
	for(int j=0; j<contours.size(); j++)
		if(pointPolygonTest(contours[j], median, false) >= 0)
			drawContours(mask, contours, j, 255, FILLED);
	
	//smooth borders
	medianBlur(mask,mask,49);
	
	//increment mask size to image size
	Mat newMask = Mat::zeros(foodImage.rows, foodImage.cols, CV_8UC1);
	for(int j=0; j<mask.rows; j++) {
		for(int k=0; k<mask.cols; k++) {
			int x = start.x+k, y=start.y+j;
			newMask.at<uchar>(y, x) = mask.at<uchar>(j, k);
		}
	}
	
	//apply new mask on full image
	image.copyTo(foodImage, newMask);
	
	return foodImage;
}


//This function generates a vector of masks, one for each plate/salad/bread.
//Since bread has its own recognition process, it's always possible to recognise it, so its mask
//will use value 13 (the bread code).
//When it is able to recognise salad (that is not always the case) it uses salad code (12) in its mask.
//Note that salad is recognizable when there are more than 2 plates (there can't be more than 2 plates,
//so the third one, that is the smallest one, must be a salad cup).
vector<Mat> generateMasks(Mat image, vector<Vec3f> plates){
	//bread management objects
	bool bread = false;
	Mat breadMask;
	
	//find salad and bread
	int saladPos = -1;
	int countPlates = 0;
	
	for(int i=0;i<plates.size();i++)
		if(plates[i][2] != 0)
			countPlates ++;
		else
			bread = true;
	
	if(countPlates == 3){
		
		//find circle with minimum radius
		int minRadius = plates[0][2];
		saladPos = 0;
		for(int i=1;i<plates.size();i++)
			if(plates[i][2] < minRadius && plates[i][2]!=0){
				minRadius = plates[i][2];
				saladPos = i;
			}
	}
	
	if(bread){
		//create bread mask as a mask of everything and then remove each plate content
		//in this way only bread pixels will be left
		breadMask = Mat::zeros(image.size(), CV_8UC1);
		for(int x=0; x<image.cols;x++){
			for(int y=0; y<image.rows; y++){
				Vec3b pixel = image.at<Vec3b>(y,x);

				if(pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0){
					breadMask.at<uchar>(y,x) = 13;
				}
			}
		}
	}
	
	//create a mask for each plate
	vector<Mat> masks;
	for(int i=0;i<plates.size();i++){
		Point center (plates[i][0], plates[i][1]);
		int radius = plates[i][2];
		int color = 255;
		
		if(radius == 0)
			//bread position: not a plate
			continue;
		
		if(i == saladPos){
			//salad often goes over the borders
			radius += 9;
			color = 12;
		}else{
			radius += 5;
		}
		
		//scan the circle and save the mask
		Mat mask = Mat::zeros(image.size(), CV_8UC1);
		for (int y = center.y - radius; y <= center.y + radius; y++) {
			for (int x = center.x - radius; x <= center.x + radius; x++) {
				if ((x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) <= radius*radius) {
					if(x>=0 && x<image.cols && y>=0 && y<image.rows){
						Vec3b pixel = image.at<Vec3b>(y,x);
						
						if(pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0){
							mask.at<uchar>(y,x) = color;
							
							if(bread)
								//remove plate content from bread mask
								breadMask.at<uchar>(y,x) = 0;
						}
					}
				}
			}
		}
		masks.push_back(mask);
	}
	
	if(bread && countNonZero(breadMask)>0){
		masks.push_back(breadMask);
	}
	return masks;
}
	