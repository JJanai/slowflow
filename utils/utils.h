/*
 * utils.h
 *
 *  Created on: Mar 22, 2016
 *      Author: jjanai
 */

#ifndef EPIC_UTILS_H_
#define EPIC_UTILS_H_

#include "../configuration.h"

#include <stdexcept>      // std::out_of_range
#include <iostream>
#include <fstream>

#include "imageLib.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdlib.h>
#include <math.h>

#include "../epic_flow_extended/image.h"
#include "../epic_flow_extended/io.h"

// include flowcode (middlebury devkit)
#include MIDDLEBURY_PATH(/colorcode.h)
#include MIDDLEBURY_PATH(/flowIO.h)

using namespace cv;
using namespace std;

#define PI 3.14159265

/*
 * Error measures
 */
double computeEPE(const image_t* flow_x, const image_t* flow_y, const image_t* gt_x, const image_t* gt_y, Mat* error_img = NULL, Mat* mask = NULL, double norm = 0);
double computeAAE(const image_t* flow_x, const image_t* flow_y, const image_t* gt_x, const image_t* gt_y, Mat* mask = NULL);
double computeRMS(const color_image_t* im1, const color_image_t* im2, const image_t* flow_x, const image_t* flow_y);

/*
* warp input image according flow
* 		scale		input 			output
* 		1.0			reference img	second img
* 		-1.0		second img		reference img
*/
void warp_image(const color_image_t* img, const image_t* flow_x, const image_t* flow_y, color_image_t* warped, float scale = 1.0f);

Mat crop(Mat src, Point center, Point extent);

void MatToCFImg(const Mat &src, CFloatImage& output);
void CFImgToMat(CFloatImage& src, Mat& output);

void cvArrow(Mat Image, int x, int y, int u, int v, CvScalar Color, int Size, int Thickness);


/*
 * Remove small segments
 */
inline uint getAddressOffsetImage(const int32_t& u,const int32_t& v,const int32_t& width) {
  return v*width+u;
}

Mat removeSmallSegments(Mat& F, float similarity_threshold, int min_segment_size);

/*
 * Bilinear interpolation for subpixel precision
 */
double bilinearInterp(double x, double y, const float* fct, int height, int width, int stride);
template<typename T>
inline static double bilinearInterp(double x, double y, const Mat& fct, int c);


/*
 * Demosaicing and raw weighting
 */
void bayer2rgb(Mat src, Mat dst, int red_x, int red_y);
void bayer2rgbGR(Mat src, Mat dst, int red_x, int red_y);

void rawWeighting(color_image_t* weights, int red_x, int red_y, float weight);

/*
 * Flow Functions
 */
void forwardBackwardCheck(Mat& mask, const Mat *forward, const Mat *backward, uint32_t FF, float epsilon, bool jetwise);
void forwardBackwardConsistency(Mat *forward_flow, Mat *backward_flow, Mat& mask, Mat& flow_diff, uint32_t FF, int S, double epsilon, bool jetwise, uint32_t skip, bool verbose = false, int threads = 1);

Mat fuseOcclusions(const Mat* forward, const Mat *occlusions, int start, int length);
Mat fuseOcclusions(image_t*** forward, const Mat *occlusions, int start, int length);

float accumulateFlow(Mat *acc_forward, const Mat *forward, const Mat &occlusions, uint32_t FF);	// using accumulated occlusion map
float accumulateFlow(Mat *acc_forward, const Mat *forward, const Mat *occlusions, uint32_t FF);
float accumulateBatches(Mat *acc_forward, Mat *acc_backward, const Mat *forward, const Mat *backward, const Mat& mask, uint32_t FF, int S, uint32_t skip, int threads = 1);
Mat accumulateConsistentBatches(Mat *acc_forward, Mat *forward_flow, Mat *backward_flow, Mat* occlusions, uint32_t FF, double epsilon, uint32_t skip, bool discard, bool verbose);

Mat flowColorImg(const image_t *w_x, const image_t *w_y, int verbose = 0, float maxrad = -1);
Mat flowColorImg(const Mat &flowf, int verbose = 0, float maxrad = -1, Mat mask = Mat());

/*
 * Read and write functions
 */
void writeToFile(Mat& mat, String file);
void readFromFile(Mat& mat, String file);
Mat readGTMiddlebury(string filename);
void writeFlowMiddlebury(Mat img, string filename);


float_image read_float_file(string filename, const int width, const int height);
void write_float_file(string filename, float_image& res, const int width, const int height);

/*
 * Derivatives
 */
void dx(const float* img, float* der, int width, int height, int stride);
void dy(const float* img, float* der, int width, int height, int stride);

template<typename T>
void mat2colorImg(const Mat& img, color_image_t *seq) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			seq->c1[i*seq->stride+j] = img.at<T>(i,j); // they use 8 bit images
			seq->c2[i*seq->stride+j] = img.at<T>(i,j);
			seq->c3[i*seq->stride+j] = img.at<T>(i,j);
		}
	}
}

template<typename T>
void mat2colorImg(const Mat& c1, const Mat& c2, const Mat& c3, color_image_t *seq) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			seq->c1[i*seq->stride+j] = c1.at<T>(i,j); // they use 8 bit images
			seq->c2[i*seq->stride+j] = c2.at<T>(i,j);
			seq->c3[i*seq->stride+j] = c3.at<T>(i,j);
		}
	}
}

template<typename T>
void colorImg2colorMat(const color_image_t *seq, Mat& img) {
	for(int i=0 ; i<seq->height ; i++) {
		for( int j=0 ; j<seq->width ; j++) {
			img.at<T>(i,j)[0] = seq->c1[i*seq->stride+j]; // they use 8 bit images
			img.at<T>(i,j)[1] = seq->c2[i*seq->stride+j];
			img.at<T>(i,j)[2] = seq->c3[i*seq->stride+j];
		}
	}
}

template<typename T>
void colorMat2colorImg(const Mat& img, color_image_t *seq) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			seq->c1[i*seq->stride+j] = img.at<T>(i,j)[0]; // they use 8 bit images
			seq->c2[i*seq->stride+j] = img.at<T>(i,j)[1];
			seq->c3[i*seq->stride+j] = img.at<T>(i,j)[2];
		}
	}
}

template<typename T>
void mat2img(const Mat& img, image_t *seq) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			seq->data[i*seq->stride+j] = img.at<T>(i,j); // they use 8 bit images
		}
	}
}

template<typename T>
void img2mat(const image_t *seq, Mat& img) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			img.at<T>(i,j) = seq->data[i*seq->stride+j] ; // they use 8 bit images
		}
	}
}

template<typename T>
double bilinearInterp(double x, double y, const Mat& fct, int c) {
    T *ptr = (T*)(fct.data);
    int width = fct.cols;
    int height = fct.rows;
    int ch = fct.channels();

    if(x < 0 || y < 0) throw out_of_range ("bilinearInterp: negative indices!");
	if(x >= width || y >= height) throw out_of_range ("bilinearInterp: out of image!");

	int y0 = (int) y;
	int x0 = (int) x;
	int y1 = y0;
	int x1 = x0;

	// interpolate in x direction
	double weight_x = 0;
	if((x0+1) < width) {
		weight_x = (x - x0);
		x1++;
	}

	// interpolate in y direction
	double weight_y = 0;
	if((y0+1) < height) {
		weight_y = (y - y0);
		y1++;
	}

	double f_x0y0 = ptr[ch * width * y0 + ch * x0 + c];
	double f_x1y0 = ptr[ch * width * y0 + ch * x1 + c];
	double f_x0y1 = ptr[ch * width * y1 + ch * x0 + c];
	double f_x1y1 = ptr[ch * width * y1 + ch * x1 + c];

	return (1 - weight_y) * (1 - weight_x) * f_x0y0 + (1 - weight_y) * weight_x * f_x1y0 + weight_y * (1 - weight_x) * f_x0y1 + weight_y * weight_x * f_x1y1;
}

#endif /* UTILS_UTILS_H_ */
