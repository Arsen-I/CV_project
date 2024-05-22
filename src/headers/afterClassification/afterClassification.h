
//File written by Alessandro Benetti (id 1210974)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <map>
#include <array>

void remaining_food_detect(cv::Mat inputImage, cv::Mat inputMask, cv::Mat& outputMask, std::vector<int> foods);

constexpr auto MAX_PIX = 256;

class Color_spectrum_layer {
	private :
		std::array<int, MAX_PIX> num_R = {0};
        std::array<int, MAX_PIX> num_G = {0};
        std::array<int, MAX_PIX> num_B = {0};
		int total_pixels = 0;
	public :
		void inc_R(unsigned char val_R);
		void inc_G(unsigned char val_G);
		void inc_B(unsigned char val_B);
		void add_RGB_pixel(unsigned char val_R, unsigned char val_G, unsigned char val_B);

		void add_quantities_RGB_spectrum(unsigned char val_R, unsigned char val_G, unsigned char val_B, int quantity_R, int quantity_G, int quantity_B);
		
		int get_R(unsigned char val_R);
		int get_G(unsigned char val_G);
		int get_B(unsigned char val_B);
		int get_total_pixels();

		void set_total_pixels(int total_pixels);
		
		void set_R(unsigned char val_R, int num_R);
		void set_G(unsigned char val_G, int num_G);
		void set_B(unsigned char val_B, int num_B);
		Color_spectrum_layer() = default;
		Color_spectrum_layer(Color_spectrum_layer& a);
		double similarity_score(Color_spectrum_layer* col_spect_layer);
		double similarity_score(unsigned char R_value, unsigned char G_value, unsigned char B_value);
		void add_Color_spectrum_layer_from_pad(const cv::Mat image, const cv::Mat mask, int pos_x, int pos_y, int pad_dim);
		//double similarity_score(cv::Mat grid_pixels);
};


class Food {
	private:
		int mask_id;
		Color_spectrum_layer* col_spect_lay;
	public:
		Food();
		Food(int mask_id);
		Food(Food& food);
		//getter and setter for Color_spectrum_layer
		Color_spectrum_layer* get_color_spectrum_layer();
		void set_color_spectrum_layer(Color_spectrum_layer& col_spec_lay);
		int get_mask_id();
		void set_mask_id(int mask_id);
};

class Plate {
	private:
		std::map<unsigned char, Food*> found_foods;
		cv::Mat mask;
		Color_spectrum_layer* col_spect_lay;
		void update_color_spectrum_layer();
	public:
		Plate();
		Plate(Plate& plate);
		Plate(const cv::Mat mask, const cv::Mat image);
		Plate(const cv::Mat mask, const cv::Mat ground_truth_mask, const cv::Mat image);
		Color_spectrum_layer* get_color_spectrum_layer();
		std::vector<Color_spectrum_layer*> get_col_spec_foods();
		std::vector<Food*> get_foods();
		cv::Mat getMask();
};

class Tray {
	private: 
		std::vector<Plate*> plates;
		cv::Mat image;
	public:
		Tray(std::vector<Plate*> vec_plates);
		Tray(Tray& tray);
		Tray(std::vector<cv::Mat> masks, cv::Mat ground_truth_mask, cv::Mat image);
		Tray(std::vector<cv::Mat> masks, cv::Mat image);
		std::vector<Plate*> getPlates();
		void detect_foods(Tray* trayBefore, cv::Mat maskAfter, int pad_dim);
		cv::Mat get_image();
		void reorder_plates();
};


/*class Hist_Img {
	public:
	//Constructor
	Dish(std::vector<cv::Mat> before_eating, std::vector<cv::Mat> after_eating);
};*/