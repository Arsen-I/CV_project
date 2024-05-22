
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//function that checks how empty or full a circle is
int checkValid(cv::Point center, int radius, cv::Mat image);

//function that delete the most overlapping circles
//if atLeastOne is true it deletes at least one circle and continue until there are less than 3
//if it is false it deletes only one circle but only if it overlaps too much
std::vector<cv::Vec3f> deleteMostOverlapping(std::vector<cv::Vec3f> circles, bool atLeastOne);

//function that keeps only the plates content or removes only it
cv::Mat togglePlatesContent(cv::Mat image, std::vector<cv::Vec3f> circles, bool remove);

//function that keep only pixels with a specific rgb value that was previously found 
cv::Mat keepBread(cv::Mat image);

//function that remove gray pixels from the image
//difference among each rgb value must be more than graynessTh
void removeGray(cv::Mat& image, int graynessTh);
