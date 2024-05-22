
//File written by Arsen Ibatullin (id 2071360)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <map>
#include <iostream>

#include "beforeClassification.h"

using namespace cv;
using namespace std;

cv::Point textPos;

cv::Mat detectMultiColorObject(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();

    // Convert the image to the HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lowerRed = cv::Scalar(4, 130, 95);
    cv::Scalar upperRed = cv::Scalar(100, 156, 175);
	cv::Scalar lowerRed1 = cv::Scalar(4, 167, 95);
    cv::Scalar upperRed1 = cv::Scalar(100, 255, 175);
    cv::Scalar lowerRed2 = cv::Scalar(4, 130, 180);
    cv::Scalar upperRed2 = cv::Scalar(100, 255, 180);
    cv::Scalar lowerPurple = cv::Scalar(10, 50, 50);
    cv::Scalar upperPurple = cv::Scalar(150, 255, 255);
    cv::Scalar lowerGreen = cv::Scalar(20, 50, 50);
    cv::Scalar upperGreen = cv::Scalar(110, 255, 255);
    cv::Scalar lowerOrange = cv::Scalar(10, 70, 150);
    cv::Scalar upperOrange = cv::Scalar(25, 255, 255);

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 5);

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, hsvImage, cv::Mat(), cv::Point(-1, -1), 2);

    // Create masks for each color
    cv::Mat redMask, redMask1, redMask2, purpleMask, greenMask, orangeMask;
    cv::inRange(hsvImage, lowerRed, upperRed, redMask);
    cv::inRange(hsvImage, lowerRed1, upperRed1, redMask1);
	cv::inRange(hsvImage, lowerRed2, upperRed2, redMask2);
    cv::inRange(hsvImage, lowerPurple, upperPurple, purpleMask);
    cv::inRange(hsvImage, lowerGreen, upperGreen, greenMask);
    cv::inRange(hsvImage, lowerOrange, upperOrange, orangeMask);

    // Combine the masks to detect the multi-color object
    cv::Mat combinedMask = redMask | redMask1 | redMask2 |purpleMask | greenMask | orangeMask;

    // Perform morphological operations to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, kernel);

    // Find contours on the combined mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the contour with the most rectangles
    int maxRectangles = 0;
    std::vector<cv::Point> maxContour;
    for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.01 * cv::arcLength(contour, true), true);
        int numRectangles = static_cast<int>(approx.size());
        if (numRectangles > maxRectangles) {
            maxRectangles = numRectangles;
            maxContour = approx;
        }
    }

    // Draw bounding rectangle and label for the contour with the most rectangles
if (!maxContour.empty()) {
    cv::Rect boundingRect = cv::boundingRect(maxContour);

    // Increase the coordinates of the bounding rectangle
    boundingRect.x -= 10;
    boundingRect.y -= 20;
    boundingRect.width += 65;
    boundingRect.height += 45;

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

    // Calculate the area of the bounding rectangle
    double boundingRectArea = boundingRect.width * boundingRect.height;

    // Check if the bounding box covers the required area
    double areaRatio = boundingRectArea / imageArea;
    if ((areaRatio >= 0.06) && (areaRatio < 0.3)) {
        cv::rectangle(image1, boundingRect, cv::Scalar(255, 255, 0), 2);

        // Add label text
        std::string text = "Salad";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), cv::Scalar(255, 255, 0), FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 12;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
    }
}
    return image1;
}

cv::Mat detectPastaWithRagu(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();
    // Convert the image to the HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lower = cv::Scalar(16, 133, 50);
    cv::Scalar upper = cv::Scalar(50, 180, 255);

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 35); 

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Image Erosion
    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Create a mask
    cv::Mat RedMask;
    cv::inRange(erodedImage, lower, upper, RedMask);

    // Find Outlines On Our Color Range Mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(RedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;
		// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.2));
    boundingRect.x -= 1*inflatePixels;
    boundingRect.y -= 1*inflatePixels;
    boundingRect.width += 2* inflatePixels;
    boundingRect.height += 2* inflatePixels;

    // Check if the rectangle occupies the necessary space from the area of the image
    if (maxArea / imageArea > 0.01) {
        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Pasta with meat sauce";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        int objectId = 3;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
    }
    

    return image1;
}

cv::Mat detectRiso(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();
    // Convert the image to the HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lower = cv::Scalar(16, 133, 128);
    cv::Scalar upper = cv::Scalar(20, 149, 190);

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 85);

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 45);

    // Image Erosion
    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Create a mask
    cv::Mat Mask;
    cv::inRange(erodedImage, lower, upper, Mask);

    // Find Outlines On Our Color Range Mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

	// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.1));
    boundingRect.x -= 1.5*inflatePixels;
    boundingRect.y -= 0.5*inflatePixels;
    boundingRect.width += 3* inflatePixels;
    boundingRect.height += 1* inflatePixels;

    // Check if the rectangle occupies the desired area of the image
    if (maxArea / imageArea > 0.01) {

        // Draw a rectangle around the object
        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Pilaw rice with peppers and peas";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 5;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
    }
    else {
        // If the rectangle is not found, run another function
        image1 = detectPastaWithRagu(image, myVector, objectDictionary);
        
    }
    
    return image1;
}

cv::Mat detectPastaWithClamsAndMussels(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();    

    // Convert the image to the HSV color space
	cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lowerRed = cv::Scalar(12, 200, 150);
    cv::Scalar upperRed = cv::Scalar(18, 240, 180);

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 35);

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Image Erosion
    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    cv::Mat RedMask;
    cv::inRange(erodedImage, lowerRed, upperRed, RedMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(RedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    double imageArea = image.rows * image.cols;

	int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.2));
    boundingRect.x -= 1.5*inflatePixels;
    boundingRect.y -= 1*inflatePixels;
    boundingRect.width += 3 * inflatePixels;
    boundingRect.height += 2 * inflatePixels;

    if (maxArea / imageArea > 0.01) {
        // Calculate the desired area based on the aspect ratio of the image
        double desiredArea = imageArea * 0.07; // 5% of the image area

        // Calculate the scale factor
        double scaleFactor = std::sqrt(desiredArea / maxArea);


        cv::rectangle(image1, boundingRect, upperRed, 2);
        // Adding text to the rectangle
        std::string text = "Pasta with clams and mussels";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upperRed, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        int objectId = 4;
        objectDictionary[objectId] = boundingRect;
        myVector.push_back(objectId);
    }
    else {
        image1 = detectRiso(image, myVector, objectDictionary);
        
    }

    return image1;
}

cv::Mat detectPastaAndPesto(cv::Mat& image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();
    // Convert the image to the HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lowerGreen = cv::Scalar(20, 85, 50); 
    cv::Scalar upperGreen = cv::Scalar(60, 255, 255); 
	
    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 155); // 155 - medianBlur

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 55);

    cv::Mat greenMask;
    cv::inRange(dilatedImage, lowerGreen, upperGreen, greenMask);

    // Find Outlines On Our Color Range Mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(greenMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

	// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.1));
    boundingRect.x -= 4.5*inflatePixels;
    boundingRect.y -= 4.5*inflatePixels;
    boundingRect.width += 9 * inflatePixels;
    boundingRect.height += 9 * inflatePixels;

    // Check if the rectangle occupies the necessary space from the area of the image
    if ((maxArea / imageArea > 0.05) && (maxArea / imageArea < 0.4)) {

        // Draw a rectangle around the object
        cv::rectangle(image1, boundingRect, upperGreen, 2);



        // Adding text to the rectangle
        std::string text = "Pasta with pesto";
		
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upperGreen, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 1;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
		
    }
    else {
        // If the rectangle is not found, run another function
        image1 = detectPastaWithClamsAndMussels(image, myVector, objectDictionary);
        
    }

	return image1;
    
}

cv::Mat detectTomatoPasta(cv::Mat& image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {

	Mat image1 = image.clone();

    // Define the lower and upper thresholds for each color in HSV
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Convert the image to the HSV color space
    cv::Scalar lowerRed = cv::Scalar(0, 145, 126); 
    cv::Scalar upperRed = cv::Scalar(10, 172, 255); 

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 155); 

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 55);

    // Image Erosion
    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 15);

    // Create a mask
    cv::Mat redMask;
    cv::inRange(erodedImage, lowerRed, upperRed, redMask);

    // Find Outlines On Our Color Range Mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(redMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

		// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.1));
    boundingRect.x -= 2*inflatePixels;
    boundingRect.y -= 2*inflatePixels;
    boundingRect.width += 4* inflatePixels;
    boundingRect.height += 4* inflatePixels;


    // Check if the rectangle occupies the necessary space from the area of the image
    if (maxArea / imageArea > 0.01) {
        cv::rectangle(image1, boundingRect, upperRed, 2);

        // Adding text to the rectangle
        std::string text = "Pasta with tomato sauce";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upperRed, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 2;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
		
    }
    else {
        // If the rectangle is not found, run another function
        image1 = detectPastaAndPesto(image, myVector, objectDictionary);
    }
    

 
return image1;
    
}

cv::Mat detectBeens(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
    Mat image1 = image.clone();

    // Define the lower and upper thresholds for each color in HSV
	cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper thresholds for each color in HSV
    cv::Scalar lowerBrown = cv::Scalar(0, 120, 80); //
    cv::Scalar upperBrown = cv::Scalar(10, 155, 172); //

    // medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 15); // -  medianBlur

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 25);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 25);


    // Create a mask for the desired color
    cv::Mat brownMask;
    cv::inRange(erodedImage, lowerBrown, upperBrown, brownMask);

    // Search for contours on the mask of the desired color
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(brownMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

	// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.1));
    boundingRect.height += 1* inflatePixels;


    // Check if the rectangle occupies more than the required area of the image
    if (maxArea / imageArea > 0.005) {
        // Draw a rectangle around our object
        cv::rectangle(image1, boundingRect, upperBrown, 2);

        // Adding text to the rectangle
        std::string text = "Beans";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upperBrown, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 10;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);

		


    }

    return image1;


}

cv::Mat detectBread(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();    

// Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Determining the range of the desired color in the HSV color space
    cv::Scalar lowerYellow = cv::Scalar(10, 35, 72); 
    cv::Scalar upperYellow = cv::Scalar(48, 92, 200); 

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 35); // 155 - medianBlur

    /// Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 5);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Create a mask for the yellow color
    cv::Mat yellowMask;
    cv::inRange(erodedImage, lowerYellow, upperYellow, yellowMask);

    // Search for contours on the yellow mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(yellowMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    /// Calculate the area of the image
    double imageArea = image.rows * image.cols;

    // Check if the rectangle occupies more than 5% of the image area
    if (maxArea / imageArea > 0.01) {
        // Draw a rectangle around the yellow object
        cv::rectangle(image1, boundingRect, upperYellow, 2);

        // Adding text to the rectangle
        std::string text = "Bread";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upperYellow, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 13;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
    }

    return image1;
}

cv::Mat detecrPotatoes(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
    Mat image1 = image.clone();

    // Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lower2 = cv::Scalar(20, 60, 175);
    cv::Scalar upper2 = cv::Scalar(34, 85, 244);

    cv::Scalar lower = cv::Scalar(20, 95, 175);
    cv::Scalar upper = cv::Scalar(34, 124, 244);

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 25);

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 12);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    // Create a mask for the yellow color
    cv::Mat Mask;
    cv::inRange(erodedImage, lower, upper, Mask);

    // Create a second mask for the yellow color
    cv::Mat Mask2;
    cv::inRange(erodedImage, lower2, upper2, Mask2);

    // Combining two masks
    cv::Mat combinedMask;
    cv::bitwise_or(Mask, Mask2, combinedMask);

    // Search for contours on the merged mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

	// Inflate the bounding rectangle
    int inflatePixels = static_cast<int>(std::round(std::max(boundingRect.width, boundingRect.height) * 0.2));
    boundingRect.x -= 1*inflatePixels;
    boundingRect.y -= 1*inflatePixels;
    boundingRect.width += 2* inflatePixels;
    boundingRect.height += 2* inflatePixels;

    // Check if the rectangle occupies more than the necessary area of the image
    if (maxArea / imageArea > 0.005) {
        // Draw a rectangle around the yellow object
        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Basil potatoes";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 11;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);

    }

    return image1;
}

cv::Mat detectFishCutlet(cv::Mat& image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();    
    // Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Determining the range of brown in the HSV color space
    cv::Scalar lower = cv::Scalar(10, 139, 139); 
    cv::Scalar upper = cv::Scalar(18, 220, 202); 

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 15); 

    // Image Dilation
    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 15);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);


    // Create a mask for the color
    cv::Mat Mask;
    cv::inRange(erodedImage, lower, upper, Mask);

    // Search for contours on the mask of the required color
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }


    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

	// Calculate the increase amount in pixels for width and height
    int increaseWidth = static_cast<int>(std::round(boundingRect.width * 1));
    int increaseHeight = static_cast<int>(std::round(boundingRect.height * 1));

    // Check if the rectangle occupies the necessary part of the image area
    if ((maxArea / imageArea > 0.0003) && (maxArea / imageArea < 0.2)) {
        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Fish cutlet";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 7;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);

    }
    return image1;


}

cv::Mat detectRabbit(cv::Mat& image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();    
	cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lower = cv::Scalar(7, 141, 78);
    cv::Scalar upper = cv::Scalar(13, 205, 167);

    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 35);

    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 25);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 15);

    cv::Mat RedMask;
    cv::inRange(erodedImage, lower, upper, RedMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(RedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }

    }

    double imageArea = image.rows * image.cols;
	// Increase the coordinates of the bounding rectangle
    boundingRect.x -= 34;
    boundingRect.y -= 54;
    boundingRect.width += 65;
    boundingRect.height += 58;
    if (maxArea / imageArea > 0.01 && maxArea / imageArea < 0.03) {

        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Rabbit";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        int objectId = 8;
        objectDictionary[objectId] = boundingRect;
        myVector.push_back(objectId);
    } else {
		image1 = detectFishCutlet(image, myVector, objectDictionary);

	}

    return image1;
}

cv::Mat detectSeafoodSalad(cv::Mat image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();   

	cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lower = cv::Scalar(0, 125, 94); 
    cv::Scalar upper = cv::Scalar(5, 164, 141); 

    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 11);

    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 15);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 15);

    cv::Mat brownMask;
    cv::inRange(noisyImage, lower, upper, brownMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(brownMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }

    double imageArea = image.rows * image.cols;

	// Increase the coordinates of the bounding rectangle
    boundingRect.x -= 20;
    boundingRect.y -= 90;
    boundingRect.width += 130;
    boundingRect.height += 125;

    if ((maxArea / imageArea > 0.00015) && (maxArea / imageArea < 0.1)) {

        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Seafood salad";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        int objectId = 9;
        objectDictionary[objectId] = boundingRect;
        myVector.push_back(objectId);
    } else {
		image1 = detectRabbit(image, myVector, objectDictionary);
		

	}

    return image1;
}

cv::Mat detectGrilledPorkCutlet(cv::Mat& image, std::vector<int>& myVector, std::map<int, cv::Rect>& objectDictionary) {
	Mat image1 = image.clone();    

// Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image1, hsvImage, cv::COLOR_BGR2HSV);

    // Determining the range of brown in the HSV color space
    cv::Scalar lower = cv::Scalar(9, 70, 121); 
    cv::Scalar upper = cv::Scalar(16, 130, 191); 

    // Add noise with medianBlur
    cv::Mat noisyImage;
    cv::medianBlur(hsvImage, noisyImage, 25); 

    cv::Mat dilatedImage;
    cv::dilate(noisyImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 15);

    cv::Mat erodedImage;
    cv::erode(dilatedImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 5);

    cv::Mat Mask;
    cv::inRange(erodedImage, lower, upper, Mask);

    // Search for contours on the brown mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Finding the Rectangular Path with the Largest Area
    double maxArea = 0;
    cv::Rect boundingRect;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            boundingRect = cv::boundingRect(contour);
        }
    }



    // Calculate the area of the image
    double imageArea = image.rows * image.cols;

    // Check if the rectangle occupies the required area of the image
    if (maxArea / imageArea > 0.01) {
        // Draw a rectangle around the brown object
        cv::rectangle(image1, boundingRect, upper, 2);

        // Adding text to the rectangle
        std::string text = "Grilled pork cutlet";
        
		cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
		Point rectangleP (0, textPos.y+5);
		rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), upper, FILLED);
        cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
        cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
		textPos.y -= (textSize.height+11);
		
        // Add the object's ID and bounding rectangle to the object dictionary
        int objectId = 6;
        objectDictionary[objectId] = boundingRect;

        // Add the object's ID to the vector
        myVector.push_back(objectId);
    } else {
		image1 = detectSeafoodSalad(image, myVector, objectDictionary);
	}

    return image1;
}

cv::Mat breadBB(cv::Mat image, cv::Mat mask, std::map<int, cv::Rect>& objectDictionary){
	//create image and mask
	cv::Mat maskCopy = mask.clone();
	threshold(maskCopy, maskCopy, 0, 255, THRESH_BINARY);
	cv::Mat image1;
	image.copyTo(image1, maskCopy);
	
	//find bounding box
	cv::Rect boundingRect = cv::boundingRect(maskCopy);
	cv::rectangle(image1, boundingRect, cv::Scalar(255,0,255), 2);
	
	//add label
	std::string text = "Bread";
	cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
	Point rectangleP (0, textPos.y+5);
	rectangle(image1, rectangleP, rectangleP + Point(textSize.width+10, -textSize.height-10), cv::Scalar(255,0,255), FILLED);
	
	//add white shadow to black text to increase contrast
	cv::putText(image1, text, textPos - Point(1,1), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2);
	cv::putText(image1, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,0,0), 2);
	textPos.y -= (textSize.height+11);
	
	return image1;
}

void manageOverlaps(cv::Mat& image, std::vector<cv::Mat>& rectMasks, std::vector<cv::Mat>& rectMasksFinal, const std::map<int, cv::Rect>& objectDictionary){
	
	
	//manage overlaps
	for(int i=0; i<rectMasks.size();i++){
		for(int j=0; j<rectMasks.size();j++){
			if(i==j)
				continue;
			
			//find ids
			double Min, Max;
			minMaxLoc(rectMasks[i], &Min, &Max);
			int id1 = Max;
			minMaxLoc(rectMasks[j], &Min, &Max);
			int id2 = Max;
			
			//compare rect i with rect j to see if they overlap
			cv::Mat intersectionMask;
			bitwise_and(rectMasks[i], rectMasks[j], intersectionMask);
			int count = countNonZero(intersectionMask);
			if(count > 0){
				if(count<countNonZero(rectMasks[i]) && count<countNonZero(rectMasks[j])){
					//the rectangles overlap a bit
					//find intersection rectangle
					cv::Rect rect1 = objectDictionary.at(id1);
					cv::Rect rect2 = objectDictionary.at(id2);
					cv::Rect intRect = rect1 & rect2;
					
					cv::Point a,b,c,d;
					a = Point(intRect.x, intRect.y);
					b = Point(intRect.x + intRect.width, intRect.y);
					c = Point(intRect.x, intRect.y + intRect.height);
					d = Point(intRect.x + intRect.width, intRect.y + intRect.height);
					
					std::pair<cv::Point, cv::Point> line;
					cv::Point p1;
					
					//reduce rects of 1 pixel (needed to find orientation)
					rect1.x++;
					rect1.width-=2;
					rect1.y++;
					rect1.height-=2;
					rect2.x++;
					rect2.width-=2;
					rect2.y++;
					rect2.height-=2;
					
					//find orientation (4 possible corner cases, 4 possible edge cases)
					if(rect1.contains(a)){ //upper-left corner
						line.first = b;
						line.second = c;
						p1 = a;
					}else if(rect1.contains(b)){ //upper-right corner
						line.first = a;
						line.second = d;
						p1 = b;
					}else if(rect1.contains(c)){ //bottom-left corner
						line.first = a;
						line.second = d;
						p1 = c;
					}else if(rect1.contains(d)){ //bottom-right corner
						line.first = b;
						line.second = c;
						p1 = d;
					}else if(rect1.contains(a + Point(0,2)) && rect1.contains(c - Point(0,2))){ //left edge
						line.first = a;
						line.second = c;
						p1 = a + Point(-2,2);
					}else if(rect1.contains(b + Point(0,2)) && rect1.contains(d - Point(0,2))){ //right edge
						line.first = b;
						line.second = d;
						p1 = b + Point(2,2);
					}else if(rect1.contains(a + Point(2,0)) && rect1.contains(b - Point(2,0))){ //upper edge
						line.first = a;
						line.second = b;
						p1 = a + Point(2,-2);
					}else if(rect1.contains(c + Point(2,0)) && rect1.contains(d - Point(2,0))){ //bottom edge
						line.first = c;
						line.second = d;
						p1 = c + Point(2,2);
					}else{ //rect1 is the biggest rect, this couple will be managed when it will be the smallest
						continue;
					}
					
					//re-increment rects of 1 pixel
					rect1.x--;
					rect1.width+=2;
					rect1.y--;
					rect1.height+=2;
					rect2.x--;
					rect2.width+=2;
					rect2.y--;
					rect2.height+=2;
					
					//extend line to full image
					double m = (double)(line.second.y-line.first.y)/(line.second.x-line.first.x);
					Point p(0,0), q(image.cols,image.rows);
					
					if((line.second.x-line.first.x)!=0){
	     				p.y = -(line.first.x - p.x) * m + line.first.y;
    	 				q.y = -(line.second.x - q.x) * m + line.second.y;
					}else{
						p.x = q.x = line.second.x;
						p.y = 0;
						q.y = image.rows;
					}
					
					line.first = p;
					line.second = q;
					
					//draw half image based on line
					cv::Mat halfed = cv::Mat::zeros(image.size(), CV_8UC1), halfedMask = cv::Mat::zeros(image.size(), CV_8UC1);
					copyMakeBorder(halfed, halfed, 1, 1, 1, 1, BORDER_CONSTANT, 255);
					cv::line(halfed, line.first, line.second, 255);
					cv::floodFill(halfedMask, halfed, p1, 255);
					
					//join masks
					bitwise_and(rectMasks[i], halfedMask, rectMasks[i]);
					bitwise_and(rectMasks[i], rectMasksFinal[i], rectMasksFinal[i]);
					bitwise_not(halfedMask, halfedMask);
					bitwise_and(rectMasks[j], halfedMask, rectMasks[j]);
					bitwise_and(rectMasks[j], rectMasksFinal[j], rectMasksFinal[j]);
				}else{
					//one of the rectangle is fully contained in the other
					
					//just put the contained after the contanitor (so it doesn't end up covering it)
					cv::Rect rect1 = objectDictionary.at(id1);
					cv::Rect rect2 = objectDictionary.at(id2);
					if(rect2.area() > rect1.area()){
						Mat tmp = rectMasksFinal[i].clone();
						rectMasksFinal[i] = rectMasksFinal[j].clone();
						rectMasksFinal[j] = tmp.clone();
					}
				}
			}
		}
	}
}

void mergePlatesAndSquares(std::vector<cv::Mat>& rectMasksFinal, std::vector<cv::Mat> plates, cv::Mat result){
	/*
	cv::Mat tmp = cv::Mat::zeros(result.size(), CV_8UC1);
 	for (Mat rectMask : rectMasksFinal) {
  		add(rectMask, tmp, tmp);
 	}
 	cv::equalizeHist(tmp, tmp);
 	imshow("tmp",tmp);
 	waitKey(0);
	//*/

	// merge plates and squares
	for (Mat rectMask : rectMasksFinal) {
		//find food code
		double Min, Max;
		minMaxLoc(rectMask, &Min, &Max);
		int objectId = Max;
		
		//find best rect-plate match
		int plate = -1, maxIntersection = -1;
		for(int i=0; i<plates.size();i++){
			cv::Mat intersection;
			bitwise_and(rectMask, plates[i], intersection);
			int count = countNonZero(intersection);
			if(maxIntersection < count){
				plate = i;
				maxIntersection = count;
			}
		}
		
		//join rectangle and plate and use objectId as mask value
		bitwise_and(rectMask, plates[plate], rectMask);
		threshold(rectMask, rectMask, 0, objectId, THRESH_BINARY);
		
		//add to the final mask
		Mat mask255;
		threshold(rectMask, mask255, 0, 255, THRESH_BINARY);
		rectMask.copyTo(result, mask255);
	}
			
	//add bread
	for(int i=0; i<plates.size(); i++){
		double Min, Max;
		minMaxLoc(plates[i], &Min, &Max);

		if(Max == 13){
			Mat mask255;
			threshold(plates[i], mask255, 0, 255, THRESH_BINARY);
			plates[i].copyTo(result, mask255);
			break;
		}
	}
}

void highlightObject(cv::Mat& image, const std::map<int, cv::Rect>& objectDictionary, std::vector<cv::Mat> plates) {
	cv::Mat result = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	
	// transform squares into masks
	std::vector<cv::Mat> rectMasks, rectMasksFinal;
	for(const auto& entry : objectDictionary){
		int objectId = entry.first;
		const cv::Rect& boundingRect = entry.second;
		cv::Mat rectMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		cv::rectangle(rectMask, boundingRect, objectId, FILLED);
		rectMasks.push_back(rectMask);
		rectMasksFinal.push_back(rectMask);
	}
	
	manageOverlaps(image, rectMasks, rectMasksFinal, objectDictionary);
	mergePlatesAndSquares(rectMasksFinal, plates, result);
	
	image = result;
}

//MAIN FUNCTION
void beforeClassify(Mat inputImage, std::vector<cv::Mat> masks, Mat& outputMask, vector<int>& foodTypes, Mat& outputBoxes){
	
	Mat inputImageCopy = inputImage.clone();
	resize(inputImageCopy,inputImageCopy,Size(),0.5,0.5,INTER_NEAREST);
	for(int i=0;i<masks.size();i++){
		resize(masks[i], masks[i], Size(), 0.5, 0.5, INTER_NEAREST);
	}

	textPos = Point(5, inputImageCopy.rows-5);
	cv::Mat copy = inputImageCopy.clone();
	cv::Mat image = inputImageCopy.clone();
	cv::Mat resultImage0 = cv::Mat::zeros(image.size(), CV_8UC3);
	std::vector<int> myVector;
	std::map<int, cv::Rect> objectDictionary;
	std::map<int, int> square;
	int breadPixelCount = 0;
	for(int i=1; i<14; i++){
		square[i] = 0;
	}
	
	for(int i=0; i<masks.size(); i++){
		double Min, Max;
		minMaxLoc(masks[i], &Min, &Max);
		Mat newMask = Mat::zeros(image.size(), CV_8UC1);

		if(Max == 13){
			resultImage0 = breadBB(image, masks[i], objectDictionary);
			
			square[13] = countNonZero(masks[i]);
			bitwise_not(masks[i], newMask);
			equalizeHist(newMask,newMask);
			Mat newImage = Mat::zeros(image.size(), CV_8UC3);
			image.copyTo(newImage, newMask);
			image = newImage.clone();
			myVector.push_back(13);
		}
	}

	// Calling a function for detecting and drawing salad

	cv::Mat resultImage1 = detectMultiColorObject(image, myVector, objectDictionary);

	cv::Mat resultImage2 = detectBeens(image, myVector, objectDictionary);

	//cv::Mat resultImage3 = detectBread(image, myVector, objectDictionary);

	cv::Mat resultImage4 = detectTomatoPasta(image, myVector, objectDictionary);

	cv::Mat resultImage5 = detecrPotatoes(image, myVector, objectDictionary);
	
	cv::Mat resultImage6 = detectGrilledPorkCutlet(image, myVector, objectDictionary);

    //cv::Mat resultImage6 = detectRabbit(image, myVector, objectDictionary);

    //cv::Mat resultImage7 = detectSeafoodSalad(image, myVector, objectDictionary);

	//cv::Mat resultImage8 = detectFishCutlet(image, myVector, objectDictionary);

	//cv::Mat resultImage9 = detectGrilledPorkCutlet(image, myVector, objectDictionary);
	


	cv::Mat combinedImage = resultImage0 | resultImage1 | resultImage2 | resultImage4 | resultImage5 | resultImage6;
	
	highlightObject(copy, objectDictionary, masks);
	
	outputBoxes = combinedImage.clone();
	
	resize(copy, copy, Size(), 2, 2, INTER_NEAREST);
	outputMask = copy.clone();

	foodTypes.assign(myVector.begin(), myVector.end());
}


//1 - pasta with pesto
//2 - pasta with tomato sauce
//3 - pasta with meat sauce
//4 - pasta with clams and mussels
//5 - pilaw rice with peppers and peas
//6 - grilled pork cutlet
//7 - fish cutlet
//8 - rabbit
//9 - seafood salad
//10 - beans
//11 - basil potatoes
//12 - salad
//13 - bread




