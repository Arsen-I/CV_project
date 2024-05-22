
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "detection.h"
#include "highLevelFunc.h"

using namespace cv;
using namespace std;

//main function that detects food and plates
void detect(Mat inputImage, Mat& outputImage, vector<Mat>& masks){
	
	//resize the images (params are choosen on resized images)
	Mat image = inputImage.clone();
	resize(image, image, Size(), 0.5, 0.5, INTER_CUBIC);
	
	Mat partialResult;
	
	//find food clusters
	outputImage = foodSelector(image, partialResult);
	
	//find plates
	vector<Vec3f> plates = platesFinder(image, outputImage);
	
	//delete everything outside the plates but mark the position of bread
	Point breadPos = breadDetector(image, plates, partialResult, outputImage);
	
	//if the bread was found select all the bread pixels and show them
	if(breadPos.x != -1){
		Vec3f breadCircle (breadPos.x, breadPos.y, 0);
		plates.push_back(breadCircle);
		
		Mat breadImage = breadSelector(image, outputImage, breadPos);
		
		//divide plates and bread
		Mat onlyBread;
		absdiff(breadImage, outputImage, onlyBread);
		
		//clean borders of plates and bread with different thresholds
		removeGray(onlyBread, 20);
		removeGray(outputImage, 40);
		
		//merge results
		add(onlyBread, outputImage, outputImage);
	
	}else //if there is no bread, clean the borders of the plates
		removeGray(outputImage, 40);
	
	masks = generateMasks(outputImage, plates);
	
	//resize back to normal
	outputImage = Mat::zeros(inputImage.size(), CV_8UC3);
	for(int i=0;i<masks.size();i++){
		resize(masks[i], masks[i], Size(), 2, 2, INTER_NEAREST);
		inputImage.copyTo(outputImage, masks[i]);
	}
	
	/*show plate circles
	Mat tmp1 = outputImage.clone();
	for(Vec3f plate : plates){
		Point center (plate[0],plate[1]);
		int radius = plate[2];
		
		circle(tmp1, center, radius, Scalar(0,255,0), 2);
		imshow("img1", tmp1);
		waitKey(0);
	}
	destroyWindow("img1");
	//*/
	
	/*show masks division
	for(Mat mask : masks){
		double Min, Max;
		minMaxLoc(mask, &Min, &Max);
		cout<<Max<<endl;
		
		if(Max == 13)
			cout<<"this is bread"<<endl;
		else if(Max == 12)
			cout<<"this is salad"<<endl;
		else
			cout<<"this is something else"<<endl;
		
		Mat tmp2 = Mat::zeros(outputImage.size(), CV_8UC3);
		outputImage.copyTo(tmp2, mask);
		imshow("img2", mask);
		
		waitKey(0);
	}
	destroyWindow("img2");
	//*/
	
	
}
