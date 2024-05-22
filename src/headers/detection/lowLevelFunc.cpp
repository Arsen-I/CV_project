
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "lowLevelFunc.h"

using namespace cv;
using namespace std;

//function that checks how empty or full a circle is
int checkValid(Point center, int radius, Mat image){
	int score = 0; //number of pixels
	int penalty = 5; //penalty for having pixels along the border
	
	//scan the circle
	for (int y = center.y - radius; y <= center.y + radius; y++) {
		for (int x = center.x - radius; x <= center.x + radius; x++) {
			if ((x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) <= radius*radius) {
				if(x>=0 && x<image.cols && y>=0 && y<image.rows){
					
					//check pixel value
					Vec3b val = image.at<Vec3b>(y,x);
					if(val[0] != 0 || val[1] != 0 || val[2] != 0)
						
						//non-empty pixel on the border
						if (abs((x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) - radius*radius) < 1000)
							score -= penalty;
						
						//non-empty pixel inside the circle
						else
							score+=2;
					
				}else
					score--; //pixel over the image borders
			}
		}
	}
	
	return (int)((score/(radius*radius*3.141597))*100);//normalize result on circle area
}

//function that delete the most overlapping circles
//if atLeastOne is true it deletes at least one circle and continue until there are less than 3
//if it is false it deletes only one circle but only if it overlaps too much
vector<Vec3f> deleteMostOverlapping(vector<Vec3f> circles, bool atLeastOne){
	vector<Vec3f> remaining;
	float maxIntersection = 0;
	int posMax = 0;
	float threshold = 0.13; //how much a circle must overlap to be deleted
	
	//scan every circle and compare it with all the others
	for(int i=0;i<circles.size();i++){
		Point center1 (cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius1 = cvRound(circles[i][2]);
		float area1 = pow(radius1,2) * 3.141592;
		float intersectionArea = 0; //total intersection area of that circle
		
		//for every other circle calculate intersection with this one and sum up
		for(int j=0;j<circles.size();j++){
			if(i!=j){
				Point center2 (cvRound(circles[j][0]), cvRound(circles[j][1]));
				int radius2 = cvRound(circles[j][2]);
				
				float distance = sqrt(pow((center1.x - center2.x),2) + pow((center1.y - center2.y),2));

				if (distance < radius1 + radius2)
					//normalize on circle[i] area
					intersectionArea += (pow((radius1 + radius2 - distance),2) * 3.141592)/area1;
			}
		}
		
		//find circle with maximum overlap
		if(intersectionArea > maxIntersection){
			maxIntersection = intersectionArea;
			posMax = i;
		}
	}
	
	
	//return options
	
	if(!atLeastOne && maxIntersection < threshold)
		return circles;
	
	if(atLeastOne || maxIntersection >= threshold){
		for(int i=0;i<circles.size();i++)
			if(i!=posMax)
				remaining.push_back(circles[i]);
	}
		
	if(remaining.size() > 3 && atLeastOne)
		return deleteMostOverlapping(remaining, true);
		
	return remaining;
}

//function that keeps only the plates content or removes only it
Mat togglePlatesContent(Mat image, vector<Vec3f> circles, bool remove){

	//the final result starts with all zeros if it must keep the plates
	Mat result = Mat::zeros(image.rows, image.cols, CV_8UC3);
	
	//or it starts with all the pixels if it must remove them
	if(remove)
		result = image.clone();
	
	//find the salad circle
	int saladPos = -1;
	
	//search for salad cup only if it's sure that there is one 
	if(circles.size() == 3){
		
		//find circle with minimum radius
		int minRadius = circles[0][2];
		saladPos = 0;
		for(int i=1;i<circles.size();i++)
			if(circles[i][2] < minRadius){
				minRadius = circles[i][2];
				saladPos = i;
			}
	}
	
	//erase or keep pixels in the plates
	for(int i=0;i<circles.size();i++){
		Point center (cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		
		//salad usually go over the border, so keep/remove a little more
		if(i==saladPos)
			radius += 7;
		
		//scan the circle
		for (int y = center.y - radius; y <= center.y + radius; y++) {
			for (int x = center.x - radius; x <= center.x + radius; x++) {
				if ((x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) <= radius*radius) {
					if(x>=0 && x<image.cols && y>=0 && y<image.rows){
						
						//remove or keep the plate content
						if(remove){
							result.at<Vec3b>(y,x)[0] = 0;
							result.at<Vec3b>(y,x)[1] = 0;
							result.at<Vec3b>(y,x)[2] = 0;
						}else{
							result.at<Vec3b>(y,x) = image.at<Vec3b>(y,x);
						}
						
					}
				}
			}
		}
		
	}
	
	return result;
	
}

//function that keep only pixels with a specific rgb value that was previously found 
Mat keepBread(Mat image){

	//rgb value
	int R = 134, G = 101, B = 69;
	
	//thresholds
	int Rth = 33, Gth = 39, Bth = 37;
	
	//go over the image
	Mat result = image.clone();
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			Vec3b val = image.at<Vec3b>(y,x);
			
			//remove pixel if not in range
			if(!(abs(val[0]-B)<Bth && abs(val[1]-G)<Gth && abs(val[2]-R)<Rth)){
				result.at<Vec3b>(y,x)[0] = 0;
				result.at<Vec3b>(y,x)[1] = 0;
				result.at<Vec3b>(y,x)[2] = 0;
			}
		}
	}
	
	return result;
}

//function that remove gray pixels from the image
//difference among each rgb value must be more than graynessTh
void removeGray(Mat& image, int graynessTh){
	for(int j=0; j<image.cols*image.rows; j++){
		Vec3b val = image.at<Vec3b>(j/image.cols, j%image.cols), zero (0,0,0);
		
		if(!(abs(val[0]-val[1])>graynessTh || abs(val[2]-val[1])>graynessTh || abs(val[0]-val[2])>graynessTh))
			image.at<Vec3b>(j/image.cols, j%image.cols) = zero;
	}
}