
//File written by Alessandro Benetti (id 1210974)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

#include "afterClassification.h"

using namespace cv;
using namespace std;




/*
Color_spectrum_layer::Color_spectrum_layer() {
	//inizialize any RGB component to 0 
	for (int i = 0; i <= 255; i++) {
		num_R[i] = 0;
		num_G[i] = 0;
		num_B[i] = 0;
	}
}
*/

Color_spectrum_layer::Color_spectrum_layer(Color_spectrum_layer& a) {
	num_R = a.num_R;
	num_G = a.num_G;
	num_B = a.num_B;
}


double Color_spectrum_layer::similarity_score(Color_spectrum_layer* col_spect_layer){
	double score = 0;
	//Variance of difference RGB spectrum
	for (int i = 0; i < 256; i++) {
		score +=	pow(((double) col_spect_layer->get_R(i)/col_spect_layer->get_total_pixels() - (double) get_R(i)/get_total_pixels()),2) +
					pow(((double) col_spect_layer->get_G(i)/col_spect_layer->get_total_pixels() - (double) get_G(i)/get_total_pixels()),2) +
					pow(((double) col_spect_layer->get_B(i)/col_spect_layer->get_total_pixels() - (double) get_B(i)/get_total_pixels()),2);
	}
	return score;
}

void Color_spectrum_layer::add_Color_spectrum_layer_from_pad(const cv::Mat image, const cv::Mat mask, int pos_x, int pos_y, int pad_dim) {
	
	for (int i = 0; (i < pad_dim) && (pos_x + i < mask.cols) ; i++)
		for (int j = 0; (j < pad_dim) && (pos_y + j < mask.rows) ; j++)
			//If the pixel is not a background
			if (mask.at<uchar>(pos_y + j, pos_x + i) != 0) {
				add_RGB_pixel(	image.at<Vec3b>(pos_y + j, pos_x + i)[0],
								image.at<Vec3b>(pos_y + j, pos_x + i)[1],
								image.at<Vec3b>(pos_y + j, pos_x + i)[2]
				);
			}

}






double Color_spectrum_layer::similarity_score(unsigned char R_value, unsigned char G_value, unsigned char B_value) {
	double score = 0;
	//Variance of difference RGB spectrum
	unsigned char R_val;
	unsigned char G_val;
	unsigned char B_val;
	for (int i = 0; i < 256; i++) {
		if (i == R_value)
			R_val = 1;
		else
			R_val = 0;
		if (i == G_value)
			G_val = 1;
		else
			G_val = 0;
		if (i == B_value)
			B_val = 1;
		else
			B_val = 0;
		score +=	pow(((double)R_val - ((double)get_R(i) / get_total_pixels())), 2) +
					pow(((double)G_val - ((double)get_G(i) / get_total_pixels())), 2) +
					pow(((double)B_val - ((double)get_B(i) / get_total_pixels())), 2);
	}
	return score;
}






/*
(cv::Mat input_image_, int filter_size_) : input_image(input_image_), filter_size(filter_size_) {
	setSize(filter_size);

	map["one"] = 1;
}

cv::Mat Filter::getResult() const {
	return result_image; // returns filter's output
}

void Filter::setSize(int size) {
	// filter_size must be odd and greater than 1 so: 3,5,7, ...
	if (size <= 2)
		filter_size = 3;
	else if (size % 2 == 0)
		filter_size = size + 1;
	else
		filter_size = size;
}

int Filter::getSize() const {
	return filter_size;
}
*/

/*
int similarity_score(Color_spectrum_layer& col_spect_lay_1, Color_spectrum_layer& col_spect_lay_2) {

	for (char c = 0; c < 256; c++)

		return 0;
}
*/

void remaining_food_detect(cv::Mat inputImageBeforeEating, cv::Mat inputImageAfterEating, cv::Mat inputMaskBeforeEating, 
	cv::Mat inputMaskAfterEating, cv::Mat& outputMask) {
	
	//, std::vector<int> foods

	std::map<unsigned char, Color_spectrum_layer*> found_food_before;
	Color_spectrum_layer* col_spect_lay;

	std::map<unsigned char, Color_spectrum_layer*> found_food_after;
	std::vector<Color_spectrum_layer*> col_spec_foods;


	//loop for cols//
	for (int x = 0; x < inputImageBeforeEating.cols; x++) {
		//loop for rows//
		for (int y = 0; y < inputImageBeforeEating.rows; y++) {
			char pixel_mask_value = inputMaskBeforeEating.at<uchar>(y, x);
			//It is not a background colour
			if (pixel_mask_value != 0) {
				//Add new layer
				if (found_food_before[pixel_mask_value] == NULL) {
					found_food_before[pixel_mask_value] = new Color_spectrum_layer();
					//col_spect_lay = new Color_spectrum_layer();
					//col_spec_foods.push_back(col_spect_lay);
				}
				else {
					//Increase the relative RGB component
					found_food_before[pixel_mask_value]->inc_R(inputImageBeforeEating.at<Vec3b>(y, x)[0]);
					found_food_before[pixel_mask_value]->inc_G(inputImageBeforeEating.at<Vec3b>(y, x)[1]);
					found_food_before[pixel_mask_value]->inc_B(inputImageBeforeEating.at<Vec3b>(y, x)[2]);
				}
			}
		}
	}





	
	
	//int num_R = col_spect_lay->get_R();
	
	//Create the sctructure to count the colour spectrum for any food found
	/*for (int food : foods) {
		//found_food[food] = true;
		col_spect_lay = new Color_spectrum_layer();
		col_spec_foods.push_back(col_spect_lay);
	}
	*/

	//loop for cols//
	for (int x = 0; x < inputMaskAfterEating.cols; x++) {
		//loop for rows//
		for (int y = 0; y < inputMaskAfterEating.rows; y++) {
			unsigned char pixel_mask_value = inputMaskAfterEating.at<uchar>(y, x);

			//It is not a background colour
			if (pixel_mask_value != 0) {
				//Add new layer
				if (found_food_after[pixel_mask_value] == NULL) {
					// found_food_after[pixel_mask_value] = col_spec_foods.back();
					// col_spec_foods.pop_back();
					found_food_after[pixel_mask_value] = new Color_spectrum_layer();
				}
				else {
					//Increase the relative RGB component
					//found_food_after[pixel_mask_value]->inc_R(inputImageAfterEating.at<Vec3b>(y, x)[0]);
					//found_food_after[pixel_mask_value]->inc_G(inputImageAfterEating.at<Vec3b>(y, x)[1]);
					//found_food_after[pixel_mask_value]->inc_B(inputImageAfterEating.at<Vec3b>(y, x)[2]);
					found_food_after[pixel_mask_value]->add_RGB_pixel(inputImageAfterEating.at<Vec3b>(y, x)[0],
						inputImageAfterEating.at<Vec3b>(y, x)[1],
						inputImageAfterEating.at<Vec3b>(y, x)[2]);
				}
			}

			
			//int x = inputImage.at<Vec3b>(10, 29)[0];
			//int y = inputImage.at<Vec3b>(10, 29)[1];
			//int z = inputImage.at<Vec3b>(10, 29)[2];
			

			//uchar pixel_mask_value = inputMask.at<uchar>(y, x);

			/*
			for (int i = 0; i < dim_kernel; i++) {
				for (int j = 0; j < dim_kernel; j++) {
					int y1 = y - middle + j;
					int x1 = x - middle + i;
					if (y1 >= 0 && y1 < image_grayscale.rows && x1 >= 0 && x1 < image_grayscale.cols) {
						if (image_grayscale.at<uchar>(y1, x1) < min_value) {
							min_value = image_grayscale.at<uchar>(y1, x1);
						}
					}
				}
			}
			image_grayscale_filtered.at<uchar>(y, x) = min_value;
			*/
		}
	}

}

void Color_spectrum_layer::add_RGB_pixel(unsigned char val_R, unsigned char val_G, unsigned char val_B) {
	inc_R(val_R);
	inc_G(val_G);
	inc_B(val_B);
	total_pixels++;
}

void Color_spectrum_layer::inc_R(unsigned char val_R) {
	num_R[val_R]++;
	//this->num_R[val_R]++;
}

void Color_spectrum_layer::inc_G(unsigned char val_G){
	num_G[val_G]++;
	//this->num_G[val_G]++;
}

void Color_spectrum_layer::inc_B(unsigned char val_B) {
	num_B[val_B]++;
	//this->num_G[val_G]++;
}

int Color_spectrum_layer::get_R(unsigned char val_R) {
	return num_R[val_R];
}

void Color_spectrum_layer::add_quantities_RGB_spectrum(unsigned char val_R, unsigned char val_G, unsigned char val_B, int quantity_R, int quantity_G, int quantity_B){
	num_R[val_R]+= quantity_R;
	num_G[val_G]+= quantity_G;
	num_B[val_B]+= quantity_B;
}



int Color_spectrum_layer::get_G(unsigned char val_G) {
	return num_G[val_G];
}

int Color_spectrum_layer::get_B(unsigned char val_B) {
	return num_B[val_B];
}

void Color_spectrum_layer::set_R(unsigned char val_R, int num_R)
{
}

int Color_spectrum_layer::get_total_pixels() {
	return total_pixels;
}

void Color_spectrum_layer::set_total_pixels(int total_pixels) {
	this->total_pixels = total_pixels;
}



Plate::Plate(){
	for (int i = 0; i < 255; i++)
		found_foods[i] = NULL;
	Food* food = new Food();
	found_foods[255] = food;
}

Plate::Plate(const Mat mask, const Mat image) : Plate(mask, mask, image) {
}

Plate::Plate(const cv::Mat mask, const cv::Mat ground_truth_mask, const cv::Mat image){
	//Set the correct mask for the plate
	this->mask = mask;
	
	bool use_ground_truth = false;
	if (mask.ptr() != ground_truth_mask.ptr())
		use_ground_truth = true;
	
	//int num_pixel = 0;
	unsigned char pixel_mask_value;

	//loop for cols//
	for (int x = 0; x < mask.cols; x++) {
		//loop for rows//
		for (int y = 0; y < mask.rows; y++) {
			if (use_ground_truth == true && (mask.at<uchar>(y, x) != 0))
				pixel_mask_value = ground_truth_mask.at<uchar>(y, x);				
			else
				pixel_mask_value = mask.at<uchar>(y, x);
			//It is not a background colour
			if (pixel_mask_value != 0) {
				//Add new layer
				if (found_foods[pixel_mask_value] == NULL) {
					// found_food_after[pixel_mask_value] = col_spec_foods.back();
					// col_spec_foods.pop_back();
					found_foods[pixel_mask_value] = new Food(pixel_mask_value);
				}
				//Increase the relative RGB component
				//check_num_B = found_foods[pixel_mask_value]->get_color_spectrum_layer()->get_B(image.at<Vec3b>(y, x)[2]);
				found_foods[pixel_mask_value]->get_color_spectrum_layer()->inc_R(image.at<Vec3b>(y, x)[0]);
				found_foods[pixel_mask_value]->get_color_spectrum_layer()->inc_G(image.at<Vec3b>(y, x)[1]);
				found_foods[pixel_mask_value]->get_color_spectrum_layer()->inc_B(image.at<Vec3b>(y, x)[2]);
				found_foods[pixel_mask_value]->get_color_spectrum_layer()->add_RGB_pixel(image.at<Vec3b>(y, x)[0],
					image.at<Vec3b>(y, x)[1],
					image.at<Vec3b>(y, x)[2]);
				//num_pixel++;
			}
		}
	}
	//cout << endl << "num_pixel : " << num_pixel << endl;
	update_color_spectrum_layer();
}





std::vector<Color_spectrum_layer*> Plate::get_col_spec_foods()
{
	std::vector<Color_spectrum_layer*> col_spec_foods;
	for (Food* food : get_foods()) {
		col_spec_foods.push_back(food->get_color_spectrum_layer());
	}
	
	return (col_spec_foods);
}

std::vector<Food*> Plate::get_foods(){
	std::vector<Food*> vect_food;
	// Get an iterator pointing to the first element in the map
	std::map<unsigned char, Food*>::iterator it = found_foods.begin();
	while (it != found_foods.end()){
		vect_food.push_back(it->second);
		++it;
	}
	return vect_food;
}

Color_spectrum_layer* Plate::get_color_spectrum_layer(){
	return col_spect_lay;
}

void Plate::update_color_spectrum_layer() {
	col_spect_lay = new Color_spectrum_layer();
	// Sum different color spectrum components
	for (int i = 0; i < 256; i++)
		for (Color_spectrum_layer* col_spect_lay_food : get_col_spec_foods()) {
			col_spect_lay->add_quantities_RGB_spectrum(i, i, i,
				col_spect_lay_food->get_R(i), col_spect_lay_food->get_G(i), col_spect_lay_food->get_B(i));
		}
	// Sum total pixels
	for (Color_spectrum_layer* col_spect_lay_food : get_col_spec_foods())
		col_spect_lay->set_total_pixels(col_spect_lay->get_total_pixels() + col_spect_lay_food->get_total_pixels());
}

Mat Plate::getMask() {
	return mask;
}


Food::Food():Food(255){
	//generic food has id = 255
	//it is when the food is not detected yet
}

Food::Food(int mask_id) {
	this->mask_id = mask_id;
	col_spect_lay = new Color_spectrum_layer();
}


Color_spectrum_layer* Food::get_color_spectrum_layer()
{
	return col_spect_lay;
}

void Food::set_color_spectrum_layer(Color_spectrum_layer& col_spec_lay){
	col_spect_lay = &col_spec_lay;
}

int Food::get_mask_id() {
	return mask_id;
}

void Food::set_mask_id(int mask_id) {
	this->mask_id = mask_id;
}




/*
waitKey(0);

//test my part
Color_spectrum_layer* col_spec_lay = new Color_spectrum_layer();


waitKey(0);

*/

Tray::Tray(vector <Plate*> vec_plates) {
	for (Plate* plate : vec_plates)
		plates.push_back(plate);
}




Tray::Tray(vector<cv::Mat> masks, cv::Mat ground_truth_mask, cv::Mat image){
	//Plate* plate_before = new Plate(masksBefore[1], before_ground_truth, detectionBefore);
	this->image = image;
	for (cv::Mat mask : masks) {
		plates.push_back(new Plate(mask, ground_truth_mask, image));
	}
}

Tray::Tray(vector<cv::Mat> masks, cv::Mat image) {
	//Plate* plate_after = new Plate(masksAfter[1], detectionAfter);
	this->image = image;
	for (cv::Mat mask : masks) {
		plates.push_back(new Plate(mask, image));
	}
}


std::vector<Plate*> Tray::getPlates() {
	return plates;
}

void Tray::reorder_plates(){
	std::vector<Plate*> new_plates;
	//Insert before the plates already detected with the food
	for (Plate* plate : plates)
		if (plate->get_foods()[0]->get_mask_id() != 255)
			new_plates.push_back(plate);
	//Insert after the plates that need to be detected
	for (Plate* plate : plates)
		if (plate->get_foods()[0]->get_mask_id() == 255)
			new_plates.push_back(plate);
	plates = new_plates;
}

void Tray::detect_foods(Tray* trayBefore, cv::Mat maskAfter, int pad_dim=1){
	std::map<double, Plate*> score_plate;
	double score_min;
	unsigned char pixel_mask_value;
	unsigned char food_id;
	Plate* plate_score_min = NULL;
	Food* food_score_min = NULL;
	std::map<Plate*, bool> detected_plates;
	cv::Mat maskAfterFood;
	
	//Reorder plates in after eating
	reorder_plates();
	
	//detect similarity in plates

	if (pad_dim <= 0 || pad_dim > maskAfter.cols || pad_dim > maskAfter.rows)
		pad_dim = 1;

	for (Plate* plate_before : trayBefore->getPlates())
		detected_plates[plate_before] = false;

	for (Plate* plate_after : getPlates()) {
		
		maskAfterFood = plate_after->getMask();

		Plate* plate_first = NULL;

		//The food is not detected yet I calculate a similarity score 
		if (plate_after->get_foods()[0]->get_mask_id() == 255) {
			score_min = 0;
			//if there is one plate in the tray
			if (trayBefore->getPlates().size() > 0) {
				//Recover the first plate that is not detected yet
				for (Plate* plate_before : trayBefore->getPlates())
					if (detected_plates[plate_before] == false) {
						plate_first = plate_before;
						break;
					}

				
				score_min = plate_after->get_foods()[0]->
					get_color_spectrum_layer()->similarity_score(
						plate_first->
						get_color_spectrum_layer()
					);
				plate_score_min = plate_first;
			}

			for (Plate* plate_before : trayBefore->getPlates()) {
				// if the plate wasn't detected before
				if (detected_plates[plate_before] == false) {
					double score = plate_after->get_foods()[0]->get_color_spectrum_layer()->
						similarity_score(plate_before->get_color_spectrum_layer());
					if (score < score_min) {
						score_min = score;
						plate_score_min = plate_before;
					}
				}
			}
	

			detected_plates[plate_score_min] = true;
			
			
				

			
			//detect similarity in foods

			if (plate_score_min != NULL) {
				//Plate_after is detected in plate_before
				//Multi-food in the plate case
				if (plate_score_min->get_foods().size() > 1) {
					//for any pixel of plate check the similarity score minimu between foods to find the before food more similar

					//loop for cols//
					for (int x = 0; x < maskAfterFood.cols; x += pad_dim) {
						//loop for rows//
						for (int y = 0; y < maskAfterFood.rows; y += pad_dim) {
							
							//Check if the pad is not a background colour
							//pixel_mask_value = maskAfterFood.at<uchar>(y, x);
							bool check_always_blackground=true;
							for(int i=0; (i<pad_dim)&&(x+i<maskAfterFood.cols)&& check_always_blackground == true; i++)
								for(int j=0; (j<pad_dim)&&(y+j<maskAfterFood.rows)&& check_always_blackground == true; j++)
									if (maskAfterFood.at<uchar>(y+j, x+i)!=0)
										check_always_blackground=false;
							
							//It is not a background colour
							//if (pixel_mask_value != 0) {

							if (check_always_blackground == false) {
															
								//Calculate on the fly the color_spectrum for the relative pad of pixels
								Color_spectrum_layer* col_spect_lay = new Color_spectrum_layer();
								col_spect_lay->add_Color_spectrum_layer_from_pad(image, maskAfterFood, x, y, pad_dim);

								score_min = 0;
								/*
								score_min = plate_score_min->get_foods()[0]->
									get_color_spectrum_layer()->similarity_score(
										image.at<Vec3b>(y, x)[0],
										image.at<Vec3b>(y, x)[1],
										image.at<Vec3b>(y, x)[2]
									);
								food_score_min = plate_score_min->get_foods()[0];
								*/								
								score_min = plate_score_min->get_foods()[0]->get_color_spectrum_layer()->similarity_score(
									col_spect_lay
									);
								food_score_min = plate_score_min->get_foods()[0];
								delete(col_spect_lay);

								for (Food* food_before : plate_score_min->get_foods()) {
									/*
									double score = food_before->get_color_spectrum_layer()->similarity_score(
										image.at<Vec3b>(y, x)[0],
										image.at<Vec3b>(y, x)[1],
										image.at<Vec3b>(y, x)[2]
									);
									*/
									Color_spectrum_layer* col_spect_lay = new Color_spectrum_layer();
									col_spect_lay->add_Color_spectrum_layer_from_pad(image, maskAfterFood, x, y, pad_dim);

									double score = food_before->get_color_spectrum_layer()->similarity_score(
										col_spect_lay
										);
									delete(col_spect_lay);
									if (score < score_min) {
										score_min = score;
										food_score_min = food_before;
									}
								}

								food_id = food_score_min->get_mask_id();

								for (int i = 0; (i < pad_dim) && (x + i < maskAfterFood.cols); i++)
									for (int j = 0; (j < pad_dim) && (y + j < maskAfterFood.rows); j++)
										if (maskAfterFood.at<uchar>(y + j, x + i) != 0) {
											maskAfterFood.at<uchar>(y + j, x + i) = food_id;
											maskAfter.at<uchar>(y + j, x + i) = food_id;
										}
							}
						}
					}



				}
				else {
					
					plate_after->get_foods()[0]->set_mask_id(plate_score_min->get_foods()[0]->get_mask_id());

					food_id = plate_after->get_foods()[0]->get_mask_id();

					//loop for cols//
					for (int x = 0; x < maskAfterFood.cols; x++) {
						//loop for rows//
						for (int y = 0; y < maskAfterFood.rows; y++) {
							pixel_mask_value = maskAfterFood.at<uchar>(y, x);
							//It is not a background colour
							if (pixel_mask_value != 0) {
								maskAfterFood.at<uchar>(y, x) = food_id;
								maskAfter.at<uchar>(y, x) = food_id;
							}
						}
					}

				}
			}
		}
		// The food is already detected correctly I copy the mask content from detection
		else {

			//Search in before plates the plate with the food already detected
			for (Plate* plate_before : trayBefore->getPlates())
				if (plate_after->get_foods()[0]->get_mask_id() == plate_before->get_foods()[0]->get_mask_id()) {
					detected_plates[plate_before] = true;
					break;
				}

			//loop for cols//
			for (int x = 0; x < maskAfterFood.cols; x++) {
				//loop for rows//
				for (int y = 0; y < maskAfterFood.rows; y++) {
					pixel_mask_value = maskAfterFood.at<uchar>(y, x);
					food_id = pixel_mask_value;
					//It is not a background colour
					if (pixel_mask_value != 0) {
						//maskAfterFood.at<uchar>(y, x) = food_id;
						maskAfter.at<uchar>(y, x) = food_id;
					}
				}
			}
		}
	}
		
}

 