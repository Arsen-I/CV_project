
//File written by Andrea Felline (id 2090597)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

cv::Mat foodSelector(cv::Mat image, cv::Mat& partialResult);

std::vector<cv::Vec3f> platesFinder(cv::Mat image, cv::Mat foodImage);

cv::Point breadDetector(cv::Mat image, std::vector<cv::Vec3f> plates, cv::Mat partialResult, cv::Mat& outputImage);

cv::Mat breadSelector(cv::Mat image, cv::Mat foodImg, cv::Point breadPos);

std::vector<cv::Mat> generateMasks(cv::Mat image, std::vector<cv::Vec3f> plates);

void removeGray(cv::Mat& image, int graynessTh);
