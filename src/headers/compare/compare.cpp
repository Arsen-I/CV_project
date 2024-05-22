
//File written by Andrea Felline (id 2090597)

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "compare.h"

using namespace std;
using namespace cv;

vector<double> compare(map<int,int> before, map<int,int> after, bool verbose){
	vector<double> scores;
	vector<string> foods = {"pasta with pesto", "pasta with tomato sauce", "pasta with meat sauce", 
					"pasta with clams and mussels",  "pilaw rice with peppers and peas",  "grilled pork cutlet", 
					"fish cutlet",  "rabbit",  "seafood salad",  "beans",  "basil potatoes",  "salad", "bread"};
	
	if(verbose)
		cout<<"List of foods scores:"<<endl;
	
	for(int food = 1; food<=13; food++){
		double score;
		
		if(before[food] == 0){
			if(after[food] != 0){
				score = -2;
				
				if(verbose)
					cout<<foods[food-1]<<": infinite"<<endl;
			}else
				score = -1;
			
		}else{
			score = (double)after[food] / (double)before[food];
			
			if(verbose)
				cout<<foods[food-1]<<": "<<score<<endl;
		}
		
		scores.push_back(score);
	}
	
	if(verbose && *max_element(begin(scores), end(scores)) == -1)
		cout<<"No food found"<<endl;
	
	return scores;
		
}

void count(map<int,int>& pixelCountingBefore, Mat maskBefore, map<int,int>& pixelCountingAfter, Mat maskAfter){
	Mat tmp;
	for(int i=1;i<=13;i++){
		tmp = Mat::zeros(maskBefore.size(), CV_8UC1);
		inRange(maskBefore, i, i, tmp);
		pixelCountingBefore[i] = countNonZero(tmp);
		
		tmp = Mat::zeros(maskBefore.size(), CV_8UC1);
		inRange(maskAfter, i, i, tmp);
		pixelCountingAfter[i] = countNonZero(tmp);
	}
	
	tmp = Mat::zeros(maskBefore.size(), CV_8UC1);
	inRange(maskBefore, 14, 255, tmp);
	if(countNonZero(tmp) > 0)
		throw invalid_argument("Masks can't have pixel with values higher than 13");

	tmp = Mat::zeros(maskBefore.size(), CV_8UC1);
	inRange(maskAfter, 14, 255, tmp);
	if(countNonZero(tmp) > 0)
		throw invalid_argument("Masks can't have pixel with values higher than 13");
}

Mat afterBB(Mat image, Mat maskAfter){
	Mat tmp, result = image.clone(), maskCopy = maskAfter.clone();
	resize(result, result, Size(), 0.5, 0.5, INTER_CUBIC);
	resize(maskCopy, maskCopy, Size(), 0.5, 0.5, INTER_NEAREST);
	
	Point textPos = Point(5, result.rows-5);
	vector<Scalar> colors = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0), 
							 Scalar(255,0,255), Scalar(0,255,255), Scalar(128,128,128), 
							 Scalar(255,0,128), Scalar(255,198,255), Scalar(64,64,128), 
							 Scalar(0,128,255), Scalar(29,166,0), Scalar(23,171,198)};
	vector<string> foods = {"Pasta with pesto", "Pasta with tomato sauce", "Pasta with meat sauce", 
					"Pasta with clams and mussels",  "Pilaw rice with peppers and peas",  "Grilled pork cutlet", 
					"Fish cutlet",  "Rabbit",  "Seafood salad",  "Beans",  "Basil potatoes",  "Salad", "Bread"};
	
	for(int i=1;i<=13;i++){
		tmp = Mat::zeros(result.size(), CV_8UC1);
		inRange(maskCopy, i, i, tmp);
		
		if(countNonZero(tmp)==0)
			continue;
		
		threshold(tmp, tmp, 0, 255, THRESH_BINARY);
		
		//add bounding box
		Rect boundingBox = boundingRect(tmp);
		rectangle(result, boundingBox, colors[i-1], 2);
		
		//add label
		Size textSize = getTextSize(foods[i-1], FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(result, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), colors[i-1], FILLED);

		//add white shadow to black text to increase contrast
		putText(result, foods[i-1], textPos - Point(1,1), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
		putText(result, foods[i-1], textPos, FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
	}
	
	return result;
}

vector<Rect> computeBB(Mat mask){
	Mat tmp;
	vector<Rect> result;
	
	for(int i=1;i<=13;i++){
		tmp = Mat::zeros(mask.size(), CV_8UC1);
		inRange(mask, i, i, tmp);
		threshold(tmp, tmp, 0, 255, THRESH_BINARY);
		Rect boundingBox;
		
		if(countNonZero(tmp)==0)
			boundingBox = Rect(-1, -1, -1, -1);
		else
			boundingBox = boundingRect(tmp);
		
		result.push_back(boundingBox);
	}
	
	return result;
}

double IoU(Rect box1, Rect box2) {
    double xA = max(box1.x, box2.x);
    double yA = max(box1.y, box2.y);
    double xB = min(box1.x + box1.width, box2.x + box2.width);
    double yB = min(box1.y + box1.height, box2.y + box2.height);
    double intersectionArea = max(0.0, xB - xA) * max(0.0, yB - yA);

    double box1Area = box1.width * box1.height;
    double box2Area = box2.width * box2.height;
    double unionArea = box1Area + box2Area - intersectionArea;

    double iou = unionArea > 0? intersectionArea / unionArea: 0;
    return iou;
}

double AP(vector<Rect> pred, vector<Rect> gt, double iouTh) {
    double TP = 0.0, FP = 0.0;
    double P = 0.0;
	
    for (Rect predBox : pred) {
        double maxIoU = 0.0;

        for (Rect gtBox : gt) {
            double iou = IoU(predBox, gtBox);
            if (iou > maxIoU) {
                maxIoU = iou;
            }
        }
		
        if(maxIoU >= iouTh)
            TP += 1;
        else
			FP += 1;
		
		P += (TP+FP) > 0? TP/(TP+FP) : 0;
    }
	
    double averagePrec = (gt.size() > 0)? P / gt.size() : 0;
    return averagePrec;
}

double mAP(vector<vector<Rect>> pred, vector<vector<Rect>> gt, double iouTh){
    double totalAP = 0.0;
    for(int i = 1; i <= 13; i++){
        double ap = AP(pred[i-1], gt[i-1], iouTh);
        totalAP += ap;
    }
	
    double mAP = totalAP / 13;
    return mAP;
}

// Funzione per calcolare l'Intersection over Union (IoU) tra due maschere.
double pixelIoU(Mat mask1, Mat mask2) {
    Mat intersection, unionMask;
    bitwise_and(mask1, mask2, intersection);
    bitwise_or(mask1, mask2, unionMask);

    double intersectionPixels = countNonZero(intersection);
    double unionPixels = countNonZero(unionMask);

    return unionPixels > 0 ? intersectionPixels/unionPixels : 0;
}

// Funzione per calcolare la mean Intersection over Union (mIoU) per ogni classe.
vector<double> mIoU(vector<Mat> predMasks, vector<Mat> gtMasks) {
    vector<double> IoUs(13, 0.0);
    vector<int> counts(13, 0);
	
	for(int j=0; j<predMasks.size(); j++){
		Mat pMask = Mat::zeros(predMasks[j].size(), CV_8UC1), gMask = Mat::zeros(gtMasks[j].size(), CV_8UC1);
	    
		for(int i = 0; i < 13; i++){
			inRange(predMasks[j], i+1, i+1, pMask);
			inRange(gtMasks[j], i+1, i+1, gMask);
			
			double IoU = pixelIoU(pMask, gMask);
			IoUs[i] += IoU;
			
			if(IoU>0)
				counts[i] ++;
		}
	}
	
	for(int i = 0; i < 13; i++)
		IoUs[i] = counts[i] > 0 ? IoUs[i] / counts[i] : 0;
	
    return IoUs;
}
