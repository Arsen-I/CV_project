
//File written by Andrea Felline (id 2090597)

#include <map>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// DESCRIPTION:
// This function compares "before" and "after" foods numbers of pixels, the compare rule is
// #A / #B
// where, for a fixed food type, #A is the number of pixels in the "after" image, while
// #B is the number of pixels in the "before" image

// INPUT:
// It takes in input two maps that assign to each food type (that is a number between 1 and 13)
// a number of pixels that is >=0. All food types should be in the map, even food types with 
// number of pixels equal to 0
// It also takes a "verbose" boolean that, if true, let the function print the results

// OUTPUT:
// Returns a vector with 13 decimal numbers: the 13 comparison scores for each food kind. Foods
// that has #B = 0 will have a -1 as score
std::vector<double> compare(std::map<int,int> before, std::map<int,int> after, bool verbose);


// DESCRIPTION:
// This function count "before" and "after" foods numbers of pixels

// INPUT:
// It takes in input two empty maps and two masks and assigns to each food type in the maps
// (that is a number between 1 and 13) a number of pixels that is >=0 (the number of pixels with
// that value in the masks.

// OUTPUT:
// Returns the two compiled maps
void count(std::map<int,int>& pixelCountingBefore, cv::Mat maskBefore, std::map<int,int>& pixelCountingAfter, cv::Mat maskAfter);


// DESCRIPTION:
// This function creates bounding boxes for "after" image

// INPUT:
// It takes in input the image with only food pixels and the after-mask

// OUTPUT:
// Returns an image with bounding boxes with half size 
cv::Mat afterBB(cv::Mat image, cv::Mat maskAfter);


// DESCRIPTION:
// This function creates bounding boxes for any image, ordering them by class

// INPUT:
// It takes in input the mask of the image

// OUTPUT:
// Returns a vector of bounding boxes, one for each class. Classes without an object in the mask will have
// bounding box = (-1,-1,-1,-1)
std::vector<cv::Rect> computeBB(cv::Mat mask);


// DESCRIPTION:
// This function calculate the mAP

// INPUT:
// It takes in input predicted and GT boxes ordered by class and a IoU treshold

// OUTPUT:
// Returns the mAP value
double mAP(std::vector<std::vector<cv::Rect>> pred, std::vector<std::vector<cv::Rect>> gt, double iouTh);


// DESCRIPTION:
// This function calculate the mIoU

// INPUT:
// It takes in input predicted and GT masks

// OUTPUT:
// Returns the mIoU value for each food
std::vector<double> mIoU(std::vector<cv::Mat> predMasks, std::vector<cv::Mat> gtMasks);