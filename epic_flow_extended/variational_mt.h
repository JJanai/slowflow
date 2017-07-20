#ifndef __VARIATIONAL_MT_H_
#define __VARIATIONAL_MT_H_

#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "image.h"
#include "array_types.h"
#include "variational_aux_mt.h"
#include "../utils/parameter_list.h"

using namespace std;
using namespace cv;

/* set flow parameters to default */
void normalize(color_image_t ** seq, u_int32_t F, ParameterList& params);

class Variational_MT {
public:
	Variational_MT() :
		one_direction(false), deriv(NULL), deriv_flow(NULL), alpha(0), delta_over3(0), gamma_over3(0), channel_w(NULL), occlusions(NULL) {

	}

	~Variational_MT() {
		if(deriv != NULL) convolution_delete(deriv);
		if(deriv_flow != NULL) convolution_delete(deriv_flow);
		if(occlusions != NULL) image_delete(occlusions);
	}

	/* Compute a refinement of the optical flow (wx and wy are modified) between im1 and im2 */
	Point2f variational(image_t *wx, image_t *wy, color_image_t * const * im, ParameterList& params);

	Point2f compute_one_level(Variational_AUX_MT& var_aux, image_t *wx, image_t *wy, color_image_t * const * im, ParameterList& params);


	void get_derivatives(color_image_t * const* im, const image_t *wx, const image_t *wy, color_image_t* w_im_s, color_image_t* w_im_sp1, image_t **mask,
			color_image_t **Ix, color_image_t **Iy, color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz,
			color_image_t **Ix_to_ref, color_image_t **Iy_to_ref, color_image_t **Iz_to_ref,
			color_image_t **Ixx_to_ref, color_image_t **Ixy_to_ref, color_image_t **Iyy_to_ref, color_image_t **Ixz_to_ref, color_image_t **Iyz_to_ref, int ref);

	/* Convert mat to color image*/
	static void mat2colorImg(Mat *src, color_image_t *dst);
	/* Convert color image to mat*/
	static void colorImg2Mat(color_image_t *src, Mat *dst);
	/* Convert image to mat*/
	static void img2Mat(const image_t *src, Mat *dst);
	static void mat2Img(const Mat* img, image_t *seq);

	void setChannelWeights(color_image_t *weights);
	image_t* getOcclusions() { return occlusions;}

	bool one_direction;
private:
	void setOmega(u_int32_t f, float a, u_int32_t F_);
	void setRho(u_int32_t f, float a, u_int32_t F_);

	convolution_t *deriv, *deriv_flow;
	float alpha, delta_over3, gamma_over3;

	vector<float> omega;
	vector<float> rho;

	color_image_t* channel_w;
    image_t *occlusions;
};

#endif
