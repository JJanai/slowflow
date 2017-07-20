#ifndef __VARIATIONAL_AUX_H_
#define __VARIATIONAL_AUX_H_

#include <stdlib.h>
#include "image.h"

#include "../penalty_functions/penalty_function.h"
#include "../penalty_functions/modified_l1_norm.h"

#include "../configuration_epic.h"

#include <vector>
#include <xmmintrin.h>
typedef __v4sf v4sf;

using namespace std;

const v4sf one = { 1, 1, 1, 1 };
const v4sf zero = { 0, 0, 0, 0 };
const float datanorm  = 0.1f*0.1f;
const float dt_scale_graphc = 0.01f;

enum robust_fcts {Robust_Color = 0, Robust_Grad = 1, Robust_Reg = 2};

class Variational_AUX_MT {
public:
	Variational_AUX_MT() :
		dt_norm(false), channel_w(NULL), robust_color(new ModifiedL1Norm(0.001f)), robust_grad(new ModifiedL1Norm(0.001f)), robust_reg(new ModifiedL1Norm(0.001f)) {
	}

	Variational_AUX_MT(int rob_fct, float rob_eps, float rob_trunc) :
		dt_norm(false), channel_w(NULL), robust_color(NULL), robust_grad(NULL), robust_reg(NULL) {
		select_robust_function(Robust_Color, rob_fct, rob_eps, rob_trunc);
		select_robust_function(Robust_Grad, rob_fct, rob_eps, rob_trunc);
		select_robust_function(Robust_Reg, rob_fct, rob_eps, rob_trunc);
	}

	~Variational_AUX_MT() {
		if(robust_color != NULL) delete robust_color;
		if(robust_grad != NULL) delete robust_grad;
		if(robust_reg != NULL) delete robust_reg;
	}

	/* compute the smoothness term */
	void compute_smoothness(int method, image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, const image_t *dpsis_weight, const image_t *dpsis_weight_x, const image_t *dpsis_weight_y,
			const convolution_t *deriv_flow, const float half_alpha) const;

	/* sub the laplacian (smoothness term) to the right-hand term */
	static void sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert);

	/* compute the successive dataterm
	   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
	   other (color) images are input */
	void add_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *du, image_t *dv,
			color_image_t **Ix, color_image_t **Iy, color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz,
			const float half_delta_over3, const float half_gamma_over3, int f, const float s) const;

	/* compute the reference dataterm
	   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
	   other (color) images are input */
	void add_data_and_match_ref(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *du, image_t *dv,
			color_image_t **Ix, color_image_t **Iy, color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz,
			const float half_delta_over3, const float half_gamma_over3, int f, const float s) const;

	/* compute local smoothness weight as a sigmoid on image gradient*/
	static image_t* compute_dpsis_weight(const color_image_t *im, float coef, const convolution_t *deriv, float mn_1 = 0, float mn_2 = 0, float mn_3 = 0, float mx_1 = 255, float mx_2 = 255, float mx_3 = 255);
	static void compute_dpsis_weight(const color_image_t *im, image_t* lum,  image_t* lum_x, image_t* lum_y, float coef, const convolution_t *deriv,
			float avg_1 = 0, float avg_2 = 0, float avg_3 = 0, float std_1 = 1, float std_2 = 1, float std_3 = 1, bool hbit = 0);

	/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
	static void image_warp(color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy, int factor);

	void optimizeOcc(image_t* occlusions, image_t **mask, color_image_t **Iz, color_image_t **Iz_to_ref, color_image_t **Ixz, color_image_t **Iyz,  color_image_t **Ixz_to_ref, color_image_t **Iyz_to_ref,
			int ref, const vector<float> rho, const vector<float> omega, float delta_over3, float gamma_over3, float penalty = 1.0, float alpha = 0.5, int graphc_it = 10);

	void optimizeFr(image_t* framerate, const image_t* occlusions, image_t **mask, color_image_t **Iz, color_image_t **Iz_to_ref, color_image_t **Ixz, color_image_t **Iyz,  color_image_t **Ixz_to_ref, color_image_t **Iyz_to_ref,
			int ref, const vector<float> rho, const vector<float> omega, float delta_over3, float gamma_over3, vector<int> wsizes, vector<float> penalties, float alpha = 0.5, int graphc_it = 10);

	// sorted wsizes in ascending order
	void optimizeOccFr(image_t* framerate, const image_t* occlusions, image_t **mask, color_image_t **Iz, color_image_t **Iz_to_ref, color_image_t **Ixz, color_image_t **Iyz,  color_image_t **Ixz_to_ref, color_image_t **Iyz_to_ref,
			int ref, const vector<float> rho, const vector<float> omega, float delta_over3, float gamma_over3, vector<int> wsizes, float alpha, float beta, float lambda_fr, float lambda_occ, int graphc_it);

	float robust_function(float val) const {
		return robust_reg->apply(val);
	}

	void select_robust_function(int who, int fct, float eps, float trunc);
	void select_robust_function(PenaltyFunction*& robust, int fct, float eps, float trunc);

	bool dt_norm;
	color_image_t *channel_w;
private:
	PenaltyFunction* robust_color;
	PenaltyFunction* robust_grad;
	PenaltyFunction* robust_reg;
};

#endif
