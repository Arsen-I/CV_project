
//File written by Andrea Felline (id 2090597), Arsen Ibatullin (id 2071360) and Alessandro Benetti (id 1210974)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "headers/detection/detection.h"
#include "headers/beforeClassification/beforeClassification.h"
#include "headers/afterClassification/afterClassification.h"
#include "headers/compare/compare.h"

using namespace cv;
using namespace std;

//test function (the body of the function is above)
void test(string datasetPath, int trays);

int main(int argc, char** argv) {
	
	//INPUT ERRORS HANDLING
	
	if(argc < 3){ //not enought arguments
		cout<<"usage:"<<endl
			<<"./project <n> <path1> [<path2>]"<<endl
			<<"where <n> is the number of tray in the dataset (put 0 to test only a couple of image before-after)"<<endl
			<<"<path1> is the path of the dataset (if n > 0) or of the 'before' image (if n = 0)"<<endl
			<<"<path2> is the path of the 'after' image (only needed if n = 0)"<<endl;
		return -1;
	}
	
	//check number of trays
	int trays;
	try{
		trays = stoi(argv[1]);
	}catch(const invalid_argument& e){ //firsst argument not a number
		cout<<"usage:"<<endl
			<<"./project <n> <path1> [<path2>]"<<endl
			<<"where <n> is the number of tray in the dataset (put 0 to test only a couple of image before-after)"<<endl
			<<"<path1> is the path of the dataset (if n > 0) or of the 'before' image (if n = 0)"<<endl
			<<"<path2> is the path of the 'after' image (only needed if n = 0)"<<endl;
		return -1;
	}
	
	if(trays > 0){ //number of tray greater than 0 -> testing
		//TESTING MODE
		
		cout<<"WARNING: dataset should be in the same format as the one given in the project pdf link (more details in the report)"<<endl<<endl;
		
		test(argv[2], trays);
		return 0;
		
	}else if(argc < 4){ //number of tray is 0 but there is only one path
		cout<<"usage:"<<endl
			<<"./project <n> <path1> [<path2>]"<<endl
			<<"where <n> is the number of tray in the dataset (put 0 to test only a couple of image before-after)"<<endl
			<<"<path1> is the path of the dataset (if n > 0) or of the 'before' image (if n = 0)"<<endl
			<<"<path2> is the path of the 'after' image (only needed if n = 0)"<<endl;
		return -1;
	}
	
	//ONE IMAGE COUPLE MODE
	
	Mat before = imread(argv[2]);
	Mat after = imread(argv[3]);
	
	if(before.data==NULL || after.data==NULL){
		cout<<"The chosen files aren't images"<<endl;
		return -2;
	}
	
	
	//detection: takes in input the image and localize the food clusters
	//returns an image with only food pixels and a vector of circles with the plates
	Mat detectionBefore, detectionAfter;
	vector<Mat> masksBefore, masksAfter;
	
	cout<<"Starting food clusters detection..."<<endl;
	
	detect(before, detectionBefore, masksBefore);
	detect(after, detectionAfter, masksAfter);
	
	cout<<"Detection ended."<<endl<<endl;
	
	
	//Classify and localize different foods in the "before" image
	Mat maskBefore, beforeBBs;
	vector<int> foodTypes;
	
	cout<<"Starting classification of 'before' image..."<<endl;
	
	beforeClassify(detectionBefore, masksBefore, maskBefore, foodTypes, beforeBBs);
	
	cout<<"'before' image classification ended..."<<endl<<endl;
	
	
	//Classify and localize "before" foods in "after" image
	cout<<"Starting classification of 'after' image..."<<endl;
	
	Tray* tray_before = new Tray(masksBefore, maskBefore, detectionBefore);
	Tray* tray_after = new Tray(masksAfter, detectionAfter);

	Mat maskAfter = Mat::zeros(detectionAfter.size(), CV_8UC1);
	tray_after->detect_foods(tray_before, maskAfter, 3);
	
	Mat afterBBs = afterBB(detectionAfter, maskAfter);
	
	cout<<"'after' image classification ended..."<<endl<<endl;
	
	
	//Compare pixel countings
	map<int,int> pixelCountingBefore = {{1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {7,0}, {8,0}, {9,0}, {10,0}, {11,0}, {12,0}, {13,0}};
	map<int,int> pixelCountingAfter = {{1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {7,0}, {8,0}, {9,0}, {10,0}, {11,0}, {12,0}, {13,0}};
	
	count(pixelCountingBefore, maskBefore, pixelCountingAfter, maskAfter);
	compare(pixelCountingBefore, pixelCountingAfter, true);
	
	
	//show results
	imshow("before", before);
	imshow("after", after);
	waitKey(0);
	
	imshow("before", beforeBBs);
	imshow("after", afterBBs);
	waitKey(0);
	
	//equalizeHist(maskBefore, maskBefore); //<-- de-comment to see better the result
	//equalizeHist(maskAfter, maskAfter); //<--de-comment to see better the result
	imshow("before", maskBefore);
	imshow("after", maskAfter);
	waitKey(0);
	
	return 0;
}


void test(string datasetPath, int trays){
	vector<vector<Rect>> pred, gt;
	vector<Mat> predMasks, gtMasks;
	
	//inizialize vectors
	for(int i=0;i<13;i++){
		vector<Rect> p, g;
		pred.push_back(p);
		gt.push_back(g);
	}
	
	//needed paths
	string trayPath = datasetPath+"/tray", gtPath = "/masks";
	
	if(datasetPath[datasetPath.size()-1] == '/')
		trayPath = datasetPath+"tray";
	
	for(int t=1;t<=trays;t++){
		cout<<"Computing tray "<<t<<"..."<<endl;
		//compute 'before' image of the tray
		string path_before = trayPath + to_string(t) + "/food_image.jpg";
		string path_before_gt = trayPath + to_string(t) + gtPath + "/food_image_mask.png";
		
		Mat before = imread(path_before.c_str());
		Mat before_gt = imread(path_before_gt.c_str());
		cvtColor(before_gt, before_gt, COLOR_BGR2GRAY);
		
		Mat detectionBefore;
		vector<Mat> masksBefore;
		detect(before, detectionBefore, masksBefore);
		
		Mat maskBefore, boundingBoxes;
		vector<int> foodTypes;
		beforeClassify(detectionBefore, masksBefore, maskBefore, foodTypes, boundingBoxes);
		
		vector<Rect> pred_before_bb = computeBB(maskBefore);
		vector<Rect> gt_before_bb = computeBB(before_gt);
		
		//save results of 'before' image
		for(int i=0;i<13;i++){
			if(pred_before_bb[i] != Rect(-1, -1, -1, -1))
				pred[i].push_back(pred_before_bb[i]);
			if(gt_before_bb[i] != Rect(-1, -1, -1, -1))
				gt[i].push_back(gt_before_bb[i]);
		}
		predMasks.push_back(maskBefore);
		gtMasks.push_back(before_gt);
		
		for(int l=1;l<=3;l++){
			//compute the 3 possible 'after' images
			string path_after = trayPath + to_string(t) + "/leftover" + to_string(l) + ".jpg";
			string path_after_gt = trayPath + to_string(t) + gtPath + "/leftover" + to_string(l) + ".png";
			
			Mat after = imread(path_after.c_str());
			Mat after_gt = imread(path_after_gt.c_str(), CV_8UC1);

			Mat detectionAfter;
			vector<Mat> masksAfter;
			detect(after, detectionAfter, masksAfter);

			Tray* tray_before = new Tray(masksBefore, maskBefore, detectionBefore);
			Tray* tray_after = new Tray(masksAfter, detectionAfter);
			Mat maskAfter = Mat::zeros(detectionAfter.size(), CV_8UC1);
			tray_after->detect_foods(tray_before, maskAfter, 3);
			
			vector<Rect> pred_after_bb = computeBB(maskAfter);
			vector<Rect> gt_after_bb = computeBB(after_gt);
			
			//save results of 'after' image
			for(int i=0;i<13;i++){
				if(pred_after_bb[i] != Rect(-1, -1, -1, -1))
					pred[i].push_back(pred_after_bb[i]);
				if(gt_after_bb[i] != Rect(-1, -1, -1, -1))
					gt[i].push_back(gt_after_bb[i]);
			}
			predMasks.push_back(maskAfter);
			gtMasks.push_back(after_gt);
		}
	}
	
	//calculate scores on all images
	cout<<"\nmAP: "<<mAP(pred, gt, 0.5)<<endl<<endl;
	
	vector<string> foods = {"Pasta with pesto", "Pasta with tomato sauce", "Pasta with meat sauce", 
					"Pasta with clams and mussels",  "Pilaw rice with peppers and peas",  "Grilled pork cutlet", 
					"Fish cutlet",  "Rabbit",  "Seafood salad",  "Beans",  "Basil potatoes",  "Salad", "Bread"};
	vector<double> mIoU_val = mIoU(predMasks, gtMasks);
	for(int i=0;i<13;i++){
		cout<<"mIoU of "<<foods[i]<<": "<<mIoU_val[i]<<endl;
	}
}
