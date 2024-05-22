
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// DESCRIPTION: 
// This function detects plates and food clusters in an image

// INPUT: 
// Takes in input an image of a tray

// OUTPUT: 
// Returns an image with '0' where there is no food and the 
// pixel original value where there is food.
// Also returns a vector of masks that can isolate the following items:
//  - Plates (the mask values will be 0 for the background and 255 for the plate pixels,
//		each plate is in a separate mask)
//  - Salad cups (the mask values will be 0 for the background and 255 or 12 for the
//		salad pixels, it depends if the algorithm is able to recognize it or not)
//  - Bread position (the mask values will be 0 for the background and 13 for the
//		bread pixels)
void detect(cv::Mat inputImage, cv::Mat& outputImage, std::vector<cv::Mat>& masks);

