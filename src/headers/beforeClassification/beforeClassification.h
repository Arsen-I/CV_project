
//File written by Arsen Ibatullin (id 2071360)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

// DESCRIPTION:
// This function classify food pixels in a "before" image

// INPUT:
// It takes in input an image that has only food pixels with value different from '0'

// OUTPUT:
// Returns an image with '0' where there is no food and a number between 1 and 13 
// that indicates the kind of food elsewhere. Returns also a vector of int with the list
// of the codes of the foods found in the tray
void beforeClassify(cv::Mat inputImage, std::vector<cv::Mat> masks, cv::Mat& outputMask, std::vector<int>& foodTypes, cv::Mat& outputBoxes);
