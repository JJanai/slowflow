#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "variational_mt.h"
#include "solver.h"
#include "../utils/utils.h"

#include <xmmintrin.h>
typedef __v4sf v4sf;

// standardize image sequence to zero mean and 255 standard deviation
void normalize(color_image_t ** seq, u_int32_t F, ParameterList& params) {
    // get ch
	u_int32_t ch = 3;

    // compute mean and std deviation of intensities for normalization
    vector<double> avg = vector<double>(ch, 0);
    vector<double> std_dev = vector<double>(ch, 0);

    // compute mean and std deviation for each channel
    for(u_int32_t f = 0; f < F; f++) {
        vector<double> avg_frame = vector<double>(ch, 0);
        vector<double> std_dev_frame = vector<double>(ch, 0);

		for(int i=0 ; i<seq[f]->height ; i++){
			for( int j=0 ; j<seq[f]->width ; j++){
				avg_frame[0] += seq[f]->c1[i*seq[f]->stride+j];
				avg_frame[1] += seq[f]->c2[i*seq[f]->stride+j];
				avg_frame[2] += seq[f]->c3[i*seq[f]->stride+j];
				std_dev_frame[0] += seq[f]->c1[i*seq[f]->stride+j]*seq[f]->c1[i*seq[f]->stride+j];
				std_dev_frame[1] += seq[f]->c2[i*seq[f]->stride+j]*seq[f]->c2[i*seq[f]->stride+j];
				std_dev_frame[2] += seq[f]->c3[i*seq[f]->stride+j]*seq[f]->c3[i*seq[f]->stride+j];
			}
		}

		avg[0] += avg_frame[0] / (seq[f]->height * seq[f]->width);
		avg[1] += avg_frame[1] / (seq[f]->height * seq[f]->width);
		avg[2] += avg_frame[2] / (seq[f]->height * seq[f]->width);

   		std_dev[0] += std_dev_frame[0] / (seq[f]->height * seq[f]->width);
   		std_dev[1] += std_dev_frame[1] / (seq[f]->height * seq[f]->width);
   		std_dev[2] += std_dev_frame[2] / (seq[f]->height * seq[f]->width);
    }

    for(u_int32_t c = 0; c < ch; c++) {
        avg[c] /= F;
        std_dev[c] = sqrt((std_dev[c] / F) - avg[c] * avg[c]) / 255.0f;
    }

    if(params.verbosity(VER_CMD))
        for(u_int32_t c = 0; c < ch; c++) {
            cout << "Intensities normalized by (I - " << avg[c] << ") / " << std_dev[c] << endl;
        }

    // normalize each channel
    for(u_int32_t f = 0; f < F; f++) {
		for(int i=0 ; i<seq[f]->height ; i++){
			for( int j=0 ; j<seq[f]->width ; j++){
				if(std_dev[0] > 0) seq[f]->c1[i*seq[f]->stride+j] = (seq[f]->c1[i*seq[f]->stride+j] - avg[0]) / std_dev[0];
				if(std_dev[1] > 0) seq[f]->c2[i*seq[f]->stride+j] = (seq[f]->c2[i*seq[f]->stride+j] - avg[1]) / std_dev[1];
				if(std_dev[2] > 0) seq[f]->c3[i*seq[f]->stride+j] = (seq[f]->c3[i*seq[f]->stride+j] - avg[2]) / std_dev[2];
   			}
   		}
    }

    stringstream avg1, avg2, avg3, std1, std2, std3;
    avg1 << avg[0];
    avg2 << avg[1];
    avg3 << avg[2];
	std1 << std_dev[0];
	std2 << std_dev[1];
	std3 << std_dev[2];

    params.insert("slow_flow_img_norm_avg_1", avg1.str(), true);
    params.insert("slow_flow_img_norm_avg_2", avg2.str(), true);
    params.insert("slow_flow_img_norm_avg_3", avg3.str(), true);
    params.insert("slow_flow_img_norm_std_1", std1.str(), true);
    params.insert("slow_flow_img_norm_std_2", std2.str(), true);
    params.insert("slow_flow_img_norm_std_3", std3.str(), true);
}

void Variational_MT::get_derivatives(color_image_t * const* im, const image_t *wx, const image_t *wy, color_image_t* w_im_s, color_image_t* w_im_sp1, image_t **mask,
				color_image_t **Ix, color_image_t **Iy, color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz,
				color_image_t **Ix_to_ref, color_image_t **Iy_to_ref, color_image_t **Iz_to_ref,
				color_image_t **Ixx_to_ref, color_image_t **Ixy_to_ref, color_image_t **Iyy_to_ref, color_image_t **Ixz_to_ref, color_image_t **Iyz_to_ref, int ref) {
	// warp all frames and compute derivatives
	int s = 0;
	if(one_direction) s = ref;

	const v4sf half = { 0.5f, 0.5f, 0.5f, 0.5f };

	for(; s < 2*ref; s++) {
		if(s < ref) {
			// warp frame s
			Variational_AUX_MT::image_warp(w_im_s, mask[s], im[s], wx, wy, s - ref);

			// warp frame s+1
			Variational_AUX_MT::image_warp(w_im_sp1, NULL, im[s + 1], wx, wy, s - ref + 1);
		} else {
			// warp frame s
			Variational_AUX_MT::image_warp(w_im_s, NULL, im[s], wx, wy, s - ref);

			// warp frame s+1
			Variational_AUX_MT::image_warp(w_im_sp1, mask[s], im[s + 1], wx, wy, s - ref + 1);
		}

		// derivatives for successive term
		color_image_t *tmp_mean = color_image_new(w_im_s->width, w_im_s->height);
		v4sf *tmp_meanp = (v4sf*) tmp_mean->c1,
			 *dtp = (v4sf*) Iz[s]->c1,
			 *im1p = (v4sf*) w_im_s->c1,
			 *im2p = (v4sf*) w_im_sp1->c1;
		for (int i = 0; i < 3 * w_im_s->height * w_im_s->stride / 4; i++) {
			// spatial derivative of mean of the warped image and reference image
			*tmp_meanp = half * ((*im2p) + (*im1p));
			// temporal derivative
			*dtp = (*im1p) - (*im2p);
			dtp += 1;
			im1p += 1; im2p += 1; tmp_meanp += 1;
		}

		color_image_convolve_hv(Ix[s], tmp_mean, deriv, NULL);
		color_image_convolve_hv(Iy[s], tmp_mean, NULL, deriv);
		color_image_convolve_hv(Ixx[s], Ix[s], deriv, NULL);
		color_image_convolve_hv(Ixy[s], Ix[s], NULL, deriv);
		color_image_convolve_hv(Iyy[s], Iy[s], NULL, deriv);
		color_image_convolve_hv(Ixz[s], Iz[s], deriv, NULL);
		color_image_convolve_hv(Iyz[s], Iz[s], NULL, deriv);

		// derivatives for reference frame
		color_image_erase(tmp_mean);
		tmp_meanp = (v4sf*) tmp_mean->c1;
		dtp = (v4sf*) Iz_to_ref[s]->c1;
		if(s < ref) {
			im1p = (v4sf*) w_im_s->c1;
			im2p = (v4sf*) im[ref]->c1;
		} else {
			im1p = (v4sf*) im[ref]->c1;
			im2p = (v4sf*) w_im_sp1->c1;
		}
		for (int i = 0; i < 3 * w_im_s->height * w_im_s->stride / 4; i++) {
			// spatial derivative of mean of the warped image and reference image
			*tmp_meanp = half * ((*im2p) + (*im1p));
			// temporal derivative
			*dtp = (*im1p) - (*im2p);
			dtp += 1;
			im1p += 1; im2p += 1; tmp_meanp += 1;
		}

		color_image_convolve_hv(Ix_to_ref[s], tmp_mean, deriv, NULL);
		color_image_convolve_hv(Iy_to_ref[s], tmp_mean, NULL, deriv);
		color_image_convolve_hv(Ixx_to_ref[s], Ix_to_ref[s], deriv, NULL);
		color_image_convolve_hv(Ixy_to_ref[s], Ix_to_ref[s], NULL, deriv);
		color_image_convolve_hv(Iyy_to_ref[s], Iy_to_ref[s], NULL, deriv);
		color_image_convolve_hv(Ixz_to_ref[s], Iz_to_ref[s], deriv, NULL);
		color_image_convolve_hv(Iyz_to_ref[s], Iz_to_ref[s], NULL, deriv);

		color_image_delete(tmp_mean);

	}
}

/* perform flow computation at one level of the pyramid */
Point2f Variational_MT::compute_one_level(Variational_AUX_MT& var_aux, image_t *wx, image_t *wy, color_image_t * const* im, ParameterList& params) {
    const int width = wx->width, height = wx->height, stride=wx->stride;

    // get parameters
    int ref = params.parameter<int>("slow_flow_S") - 1;

    int smoothing = params.parameter<int>("slow_flow_smoothing", "0");

    int alter_it = params.parameter<int>("slow_flow_niter_alter", "1");
    int outer_it = params.parameter<int>("slow_flow_niter_outer");

    int inner_it = params.parameter<int>("slow_flow_niter_inner");
    int solver_it = params.parameter<int>("slow_flow_niter_solver");
    int graphc_it = params.parameter<int>("slow_flow_niter_graphc", "10");

    float outer_thres = params.parameter<float>("slow_flow_thres_outer");
    float inner_thres = params.parameter<float>("slow_flow_thres_inner");
    float solver_omega = params.parameter<float>("slow_flow_sor_omega");

    bool occlusion_reasoning = params.parameter<bool>("slow_flow_occlusion_reasoning","0");
    float occlusion_penalty = params.parameter<float>("slow_flow_occlusion_penalty","1.0");
    float occlusion_alpha = params.parameter<float>("slow_flow_occlusion_alpha","0.5");

    float hbit = params.parameter<bool>("16bit","0");

    image_t *du = image_new(width,height), *dv = image_new(width,height),  		// the flow increment
    		*old_du = image_new(width,height), *old_dv = image_new(width,height),										// increments from last iteration
        *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), 								// horiz: (i,j) contains the diffusivity coeff from (i,j) to (i+1,j)
        *uu = image_new(width,height), *vv = image_new(width,height), 													// flow plus flow increment
        *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), 				// system matrix A of Ax=b for each pixel
        *b1 = image_new(width,height), *b2 = image_new(width,height); 													// system matrix b of Ax=b for each pixel
    image_t **mask = new image_t*[2 * ref];																				// mask containing 0: outside image boundary, occluded or turned off by temporal window, or a weighting according the win size


	if(occlusions != NULL) image_delete(occlusions);
	occlusions = image_new(width,height);																	// window size for each pixel

    color_image_t *w_im_s = color_image_new(width,height), *w_im_sp1 = color_image_new(width,height), 								// warped images
        **Ix = new color_image_t*[2 * ref], **Iy = new color_image_t*[2 * ref], **Iz = new color_image_t*[2 * ref], 		// first order derivatives
        **Ixx = new color_image_t*[2 * ref], **Ixy = new color_image_t*[2 * ref], **Iyy = new color_image_t*[2 * ref],  // second order derivatives
		**Ixz = new color_image_t*[2 * ref], **Iyz = new color_image_t*[2 * ref];

    color_image_t **Iz_to_ref = new color_image_t*[2 * ref], **Ix_to_ref = new color_image_t*[2 * ref],	**Iy_to_ref = new color_image_t*[2 * ref],			// first order derivatives
    	 **Ixx_to_ref = new color_image_t*[2 * ref], **Ixy_to_ref = new color_image_t*[2 * ref], **Iyy_to_ref = new color_image_t*[2 * ref],				// second order derivatives
         **Ixz_to_ref = new color_image_t*[2 * ref], **Iyz_to_ref = new color_image_t*[2 * ref];

    // init with zero
    image_erase(occlusions);

    // init occlusion
    if(one_direction || occlusion_reasoning)
    	fill_n(occlusions->data, occlusions->height*occlusions->stride, -1.0);												// init with occlusion in the past (only use forward terms)

    // data norm for each framerate if enabled
    float data_norm = 0;
	for(int s = 0; s < ref; s++)  {
		data_norm += rho[s] + omega[s];
	}

    // initialize each derivative array
    for(int s = 0; s < (2 * ref); s++) {
    	mask[s] = image_new(width,height);
    	Ix[s] = color_image_new(width,height);
    	Iy[s] = color_image_new(width,height);
    	Iz[s] = color_image_new(width,height);
    	Ixz[s] = color_image_new(width,height);
    	Iyz[s] = color_image_new(width,height);
    	Ixx[s] = color_image_new(width,height);
    	Ixy[s] = color_image_new(width,height);
    	Iyy[s] = color_image_new(width,height);
    	Iz_to_ref[s] = color_image_new(width,height);
    	Ix_to_ref[s] = color_image_new(width,height);
    	Iy_to_ref[s] = color_image_new(width,height);
    	Ixx_to_ref[s] = color_image_new(width,height);
    	Ixy_to_ref[s] = color_image_new(width,height);
    	Iyy_to_ref[s] = color_image_new(width,height);
    	Ixz_to_ref[s] = color_image_new(width,height);
    	Iyz_to_ref[s] = color_image_new(width,height);
    }
  
	// get maximum and minimum intensities of the channels
	float avg_1 = params.parameter<double>("slow_flow_img_norm_avg_1", "0"), avg_2 = params.parameter<double>("slow_flow_img_norm_avg_2", "0"), avg_3 = params.parameter<double>("slow_flow_img_norm_avg_3", "0"),
			std_1 = params.parameter<double>("slow_flow_img_norm_std_1", "1"), std_2 = params.parameter<double>("slow_flow_img_norm_std_2", "1"), std_3 = params.parameter<double>("slow_flow_img_norm_std_3", "1");

    // use preset local smooth weights or compute
	image_t *dpsis_weight = image_new(width,height);
    image_t *dpsis_weight_x = image_new(width,height);
    image_t *dpsis_weight_y = image_new(width,height);
    Variational_AUX_MT::compute_dpsis_weight(im[ref], dpsis_weight, dpsis_weight_x, dpsis_weight_y, 5.0, deriv, avg_1, avg_2, avg_3, std_1, std_2, std_3, hbit); 		// image gradient for weighting

	// initialize uu and vv with wx and wy
	memcpy(uu->data,wx->data,wx->stride*wx->height*sizeof(float));
	memcpy(vv->data,wy->data,wy->stride*wy->height*sizeof(float));

	Point2f avg_change = Point2f(0,0);
    for(int alter_iteration = 0 ; alter_iteration < alter_it; alter_iteration++) {
    	// warping and compute derivatives for discrete optimizations (replaces first warping in outer loop)
		get_derivatives(im, wx, wy, w_im_s, w_im_sp1, mask, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, Ix_to_ref, Iy_to_ref, Iz_to_ref, Ixx_to_ref, Ixy_to_ref, Iyy_to_ref, Ixz_to_ref, Iyz_to_ref, ref);

		// discrete optimization before outer loop of continuous optimization
		if(alter_iteration > 0) {
			// optimize occlusion variables only
			if(occlusion_reasoning && !one_direction)
				var_aux.optimizeOcc(occlusions, mask, Iz, Iz_to_ref, Ixz, Iyz, Ixz_to_ref, Iyz_to_ref, ref, rho, omega, delta_over3, gamma_over3, occlusion_penalty, occlusion_alpha, graphc_it);

			// output occlusion variables
			if(occlusion_reasoning && params.exists("slow_flow_occlusions_output")) {
				Mat vis_occ(occlusions->height, occlusions->width, CV_32FC1);
				img2Mat(occlusions, &vis_occ);
				vis_occ = (vis_occ + 1) * 0.5f;
				vis_occ.convertTo(vis_occ, CV_8UC1, 255);

				stringstream occF;
				occF << params.parameter("slow_flow_occlusions_output") << alter_iteration << ".png";
				imwrite((occF.str()), vis_occ);
			}
		}

		for(int i_outer_iteration = 0 ; i_outer_iteration < outer_it; i_outer_iteration++) {
			// warp images and get derivatives
			if(i_outer_iteration > 0)
				get_derivatives(im, wx, wy, w_im_s, w_im_sp1, mask, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, Ix_to_ref, Iy_to_ref, Iz_to_ref, Ixx_to_ref, Ixy_to_ref, Iyy_to_ref, Ixz_to_ref, Iyz_to_ref, ref);

			// set mask according to occlusion variables and framerate variables
			v4sf *occ = (v4sf*) occlusions->data;
			for(int i=0 ; i<height*stride/4; i++) {
				v4sf factor = _mm_and_ps(_mm_cmpeq_ps(*occ, zero), one);

				factor[0] = (1 + factor[0]) * data_norm;
				factor[1] = (1 + factor[1]) * data_norm;
				factor[2] = (1 + factor[2]) * data_norm;
				factor[3] = (1 + factor[3]) * data_norm;

				v4sf backward = _mm_and_ps(_mm_cmpge_ps(*occ, zero), one) / factor;	//
				v4sf forward = _mm_and_ps(_mm_cmple_ps(*occ, zero), one) / factor;

				int s = 0;
				if(one_direction) s = ref;

				for(; s < 2*ref; s++) {
					v4sf turnoff = one;
					v4sf *m = (v4sf*) &mask[s]->data[4*i];

					// turn off frames that are occluded!
					if(s < ref)
						(*m) = turnoff * backward * (*m);
					else
						(*m) = turnoff * forward * (*m);
				}

				occ+=1;
			}

			// erase du and dv
			image_erase(du);
			image_erase(dv);

			// inner fixed point iterations
			for(int i_inner_iteration = 0 ; i_inner_iteration < inner_it; i_inner_iteration++) {
				// store old du dv
				memcpy(old_du->data,du->data,du->stride*du->height*sizeof(float));
				memcpy(old_dv->data,dv->data,dv->stride*dv->height*sizeof(float));

				//  compute robust function for smoothness term
				var_aux.compute_smoothness(smoothing, smooth_horiz, smooth_vert, uu, vv, dpsis_weight, dpsis_weight_x, dpsis_weight_y, deriv_flow, alpha);

				// reset system matrix to 0
				image_erase(a11);
				image_erase(a12);
				image_erase(a22);
				image_erase(b1);
				image_erase(b2);

				// sum data term and gradient constancy term over all frames
				for(int s = 0; s < ref; s++) {
					// past data terms
					if(!one_direction) {
						// successive data term
						if(rho[ref - 1 - s] > 0)
							var_aux.add_data_and_match(a11, a12, a22, b1, b2, mask[s], du, dv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, rho[ref - 1 - s] * delta_over3, rho[ref - 1 - s] * gamma_over3, s, s - ref);
						// reference data term
						if(omega[ref - 1 - s] > 0)
							var_aux.add_data_and_match_ref(a11, a12, a22, b1, b2, mask[s], du, dv, Ix_to_ref, Iy_to_ref, Iz_to_ref, Ixx_to_ref, Ixy_to_ref, Iyy_to_ref, Ixz_to_ref, Iyz_to_ref, omega[ref - 1 - s] * delta_over3, omega[ref - 1 - s] * gamma_over3, s, s - ref);
					}

					// future data terms
					// successive data term
					if(rho[s] > 0)
						var_aux.add_data_and_match(a11, a12, a22, b1, b2, mask[ref + s], du, dv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, rho[s] * delta_over3, rho[s] * gamma_over3, ref + s, s);
					// reference data term
					if(omega[s] > 0)
						var_aux.add_data_and_match_ref(a11, a12, a22, b1, b2, mask[ref + s], du, dv, Ix_to_ref, Iy_to_ref, Iz_to_ref, Ixx_to_ref, Ixy_to_ref, Iyy_to_ref, Ixz_to_ref, Iyz_to_ref, omega[s] * delta_over3, omega[s] * gamma_over3, ref + s, s + 1);
				}

				// second derivative of smoothness term
				Variational_AUX_MT::sub_laplacian(b1, uu, smooth_horiz, smooth_vert);
				Variational_AUX_MT::sub_laplacian(b2, vv, smooth_horiz, smooth_vert);

				// solve system
				sor_coupled(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, solver_it, solver_omega);

				// update flow plus flow increment
				v4sf *uup = (v4sf*) uu->data, *vvp = (v4sf*) vv->data,
					 *wxp = (v4sf*) wx->data, *wyp = (v4sf*) wy->data,
					 *dup = (v4sf*) du->data, *dvp = (v4sf*) dv->data,
					 *odup = (v4sf*) old_du->data, *odvp = (v4sf*) old_dv->data;

				float avg_change_du = 0, avg_change_dv = 0;
				for(int i=0 ; i<height*stride/4; i++) {
					// set all unused elements to zero
					int ignore = ((i*4 + 3) % stride) + 1 - width;
					if(ignore > 0) {
						ignore = min(ignore, 4);
						for(int j = 0; j < ignore; j++) {
							(*dup)[3 - j] = 0;
							(*dvp)[3 - j] = 0;
						}
					}

					// compute average change du and dv
					v4sf diff = ((*odup) - (*dup));
					avg_change_du += fabs((float) diff[0]) + fabs((float) diff[1]) + fabs((float) diff[2]) + fabs((float) diff[3]);

					diff = ((*odvp) - (*dvp));
					avg_change_dv += fabs((float) diff[0]) + fabs((float) diff[1]) + fabs((float) diff[2]) + fabs((float) diff[3]);

					// update uu
					(*uup) = (*wxp) + (*dup);
					(*vvp) = (*wyp) + (*dvp);
					uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;odup+=1;odvp+=1;
				}

				avg_change_du /= (height*width);
				avg_change_dv /= (height*width);

				if(params.verbosity(VER_CMD))
					cout << "	inner it " << i_inner_iteration << "	avg change " << avg_change_du << "," << avg_change_dv<< endl;

				if(max(avg_change_du, avg_change_dv) < inner_thres)
					break;
			}

			// compute average change du and dv
			v4sf *uup = (v4sf*) uu->data, *vvp = (v4sf*) vv->data, *wxp = (v4sf*) wx->data, *wyp = (v4sf*) wy->data;
			float avg_change_wx = 0, avg_change_wy = 0;
			for(int i=0 ; i<height*stride/4 ; i++) {
				v4sf diff = ((*uup) - (*wxp));
				avg_change_wx += fabs((float) diff[0]) + fabs((float) diff[1]) + fabs((float) diff[2]) + fabs((float) diff[3]);

				diff = ((*vvp) - (*wyp));
				avg_change_wy += fabs((float) diff[0]) + fabs((float) diff[1]) + fabs((float) diff[2]) + fabs((float) diff[3]);

				uup+=1; vvp+=1; wxp+=1; wyp+=1;
			}

			avg_change_wx /= (height*width);
			avg_change_wy /= (height*width);

			// add flow increment to current flow
			memcpy(wx->data,uu->data,uu->stride*uu->height*sizeof(float));
			memcpy(wy->data,vv->data,vv->stride*vv->height*sizeof(float));

			if(params.verbosity(VER_CMD))
				cout << "outer it " << i_outer_iteration << "	avg change " << avg_change_wx << "," << avg_change_wy << endl;

			avg_change = Point2f(avg_change_wx, avg_change_wy);

			if(max(avg_change_wx, avg_change_wy) < outer_thres)
				break;
		}
    }

    if(params.verbosity(VER_CMD))
    	std::cout << std::endl;

    // free memory
    image_delete(du); image_delete(dv);
    image_delete(old_du); image_delete(old_dv);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete(a11); image_delete(a12); image_delete(a22);
    image_delete(b1); image_delete(b2);
    image_delete(dpsis_weight);
    image_delete(dpsis_weight_x);
    image_delete(dpsis_weight_y);

	color_image_delete(w_im_s);
	color_image_delete(w_im_sp1);

    for(int s = 0; s < (2 * ref); s++) {
        image_delete(mask[s]);

		color_image_delete(Ix[s]); color_image_delete(Iy[s]); color_image_delete(Iz[s]);
		color_image_delete(Ixx[s]); color_image_delete(Ixy[s]); color_image_delete(Iyy[s]); color_image_delete(Ixz[s]); color_image_delete(Iyz[s]);

		color_image_delete(Iz_to_ref[s]);
		color_image_delete(Ix_to_ref[s]);
		color_image_delete(Iy_to_ref[s]);
		color_image_delete(Ixx_to_ref[s]);
		color_image_delete(Ixy_to_ref[s]);
		color_image_delete(Iyy_to_ref[s]);
		color_image_delete(Ixz_to_ref[s]);
		color_image_delete(Iyz_to_ref[s]);
    }

    delete[] mask;
    delete[] Ix;
    delete[] Iy;
    delete[] Iz;
    delete[] Ixx;
    delete[] Ixy;
    delete[] Iyy;
    delete[] Ixz;
    delete[] Iyz;
    delete[] Iz_to_ref;
    delete[] Ix_to_ref;
    delete[] Iy_to_ref;
    delete[] Ixx_to_ref;
    delete[] Ixy_to_ref;
    delete[] Iyy_to_ref;
    delete[] Ixz_to_ref;
    delete[] Iyz_to_ref;

    return avg_change;
}

void Variational_MT::setOmega(u_int32_t f, float a, u_int32_t F_) {
    // check if f is valid
    if(f >= F_)
    	return;

    // init all data term weights with 1.0
    if(omega.size() < F_)
    	omega.resize(F_, 1.0);

    // set data term weight f to a
    omega[f] = a;
}

void Variational_MT::setRho(u_int32_t f, float a, u_int32_t F_) {
    // check if f is valid
    if(f >= F_)
    	return;

    // init all data term weights with 1.0
    if(rho.size() < F_)
    	rho.resize(F_, 1.0);

    // set data term weight f to a
    rho[f] = a;
}

void Variational_MT::setChannelWeights(color_image_t *weights) {
	channel_w = weights;
}

/* Compute a refinement of the optical flow (wx and wy are modified) between im1 and im2 */
Point2f Variational_MT::variational(image_t *wx, image_t *wy, color_image_t * const* im, ParameterList& params){
	params.insert("final", "0", true);

	Point2f avg_change = Point2f(0,0);

    // set parameters
	Variational_AUX_MT var_aux;
	var_aux.dt_norm = params.parameter<bool>("slow_flow_dataterm", "1");		// turn on/off dataterm normalization
	if(channel_w != NULL)
		var_aux.channel_w = channel_w;
	else {
		var_aux.channel_w = color_image_new(wx->width, wx->height);
    	fill_n(var_aux.channel_w->c1, 3*wx->height*wx->stride, 1.0);												// set channel weights to 1
	}

    int ref = params.parameter<int>("slow_flow_S") - 1;
    int F = 2*ref + 1;
	int height = im[0]->height, filter_size;
	int L = params.parameter<int>("slow_flow_layers");
	float p_scale = params.parameter<float>("slow_flow_p_scale");

    alpha = params.parameter<float>("slow_flow_alpha");
    gamma_over3 = params.parameter<float>("slow_flow_gamma") / 3.0f;
    delta_over3 = params.parameter<float>("slow_flow_delta") / 3.0f;

	if(params.exists("slow_flow_method") && params.parameter("slow_flow_method").compare("forward") == 0)
		one_direction = true;

    // set robust function
	var_aux.select_robust_function(Robust_Color, params.parameter<int>("slow_flow_robust_color"), params.parameter<float>("slow_flow_robust_color_eps"), params.parameter<float>("slow_flow_robust_color_truncation"));
	if(params.exists("slow_flow_robust_grad")) var_aux.select_robust_function(Robust_Grad, params.parameter<int>("slow_flow_robust_grad"), params.parameter<float>("slow_flow_robust_grad_eps"), params.parameter<float>("slow_flow_robust_grad_truncation"));
	else var_aux.select_robust_function(Robust_Grad, params.parameter<int>("slow_flow_robust_color"), params.parameter<float>("slow_flow_robust_color_eps"), params.parameter<float>("slow_flow_robust_color_truncation"));
	var_aux.select_robust_function(Robust_Reg, params.parameter<int>("slow_flow_robust_reg"), params.parameter<float>("slow_flow_robust_reg_eps"), params.parameter<float>("slow_flow_robust_reg_truncation"));

    // set weights for reference data term and successive data term
	for(int a = 0; a < ref; a++) {
		stringstream streamO;
		streamO << "slow_flow_omega_" << a;
		setOmega(a, params.parameter<float>(streamO.str(), "1.0"), ref);
		stringstream streamR;
		streamR << "slow_flow_rho_" << a;
		setRho(a, params.parameter<float>(streamR.str(), "1.0"), ref);
	}

    float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
    deriv = convolution_new(2, deriv_filter, 0);
    float deriv_filter_flow[2] = {0.0f, -0.5f};
    deriv_flow = convolution_new(1, deriv_filter_flow, 0);

	// create pyramid
	color_image_t*** pyramid = new color_image_t**[L];

	float sigma = 1/sqrt(2*p_scale);
	float *presmooth_filter = gaussian_filter(sigma, &filter_size);
	convolution_t *presmoothing = convolution_new(filter_size, presmooth_filter, 1);


    for(int l = 0; l < L; l++) {
    	pyramid[l] = new color_image_t*[F];

		// smooth and rescale each image of a layer
		for(int s = 0; s < F; s++) {
			if(l == 0) {
				pyramid[l][s] = color_image_new(im[s]->width, im[s]->height);
				if(params.parameter<float>("sigma", "0") > 0) {
					float *smooth_filter = gaussian_filter(params.parameter<float>("slow_flow_sigma"), &filter_size);
					convolution_t *smoothing = convolution_new(filter_size, smooth_filter, 1);

					color_image_convolve_hv(pyramid[l][s], im[s], smoothing, smoothing);

					convolution_delete(smoothing);
					free(smooth_filter);
				} else {
			        memcpy(pyramid[l][s]->c1, im[s]->c1, 3 * im[s]->stride*im[s]->height*sizeof(float));
			        pyramid[l][s]->c2 = pyramid[l][s]->c1 + pyramid[l][s]->stride*height;
			        pyramid[l][s]->c3 = pyramid[l][s]->c2 + pyramid[l][s]->stride*height;
				}
			} else {
				Mat tmp(pyramid[l - 1][s]->height, pyramid[l - 1][s]->width, CV_32FC3);
				colorImg2Mat(pyramid[l - 1][s], &tmp);

				GaussianBlur(tmp, tmp, Size(0, 0), sigma, sigma, BORDER_REPLICATE);

				float nw = floor(pyramid[l - 1][s]->width * p_scale);
				float nh = floor(pyramid[l - 1][s]->height * p_scale);
				resize(tmp, tmp, Size(nw, nh), 0, 0, INTERP_LINEAR); // bilinear for subpixel values since we downscale with 0.9

				pyramid[l][s] = color_image_new(tmp.cols, tmp.rows);
				mat2colorImg(&tmp, pyramid[l][s]);
			}
		}

		// output smoothed and rescaled image if verbose > 1
		if(params.verbosity(VER_IMG_PYR)) {
			Mat tmp_mat = Mat(pyramid[l][0]->height, pyramid[l][0]->width, CV_32FC3);

			// convert to mat
			colorImg2Mat(pyramid[l][0], &tmp_mat);

			// histogram normalization 0..255
			vector<Mat> tmp_channels(3);
			split(tmp_mat, tmp_channels);
			for(uint32_t i = 0; i < tmp_channels.size(); i++) {
				double min, max;
				cv::minMaxLoc(tmp_channels[i], &min, &max);
				tmp_channels[i] = 255 * (tmp_channels[i] - min) / (max  - min);
			}
			merge(tmp_channels, tmp_mat);
			tmp_mat.convertTo(tmp_mat, CV_8UC3, 1);

			// show image
			stringstream title;
			title << "Image " << 0 << " layer " << L - 1 - l;
			namedWindow( title.str(), WINDOW_FREERATIO );               // Create a window for display.
			moveWindow(title.str(), 100, 100);
			resizeWindow(title.str(), pyramid[0][0]->width, pyramid[0][0]->height);
			imshow( title.str(), tmp_mat );                            // Show our image inside it.
			waitKey(0);
		}

		// break if image too small
		if(floor(pyramid[l][0]->width * p_scale) <= presmoothing->order + 1 || floor(pyramid[l][0]->height * p_scale) <= presmoothing->order + 1) {
			cerr << "Maximum number of layers reached (" << l << ") because of convolution of order " << presmoothing->order << "!" << endl;
			L = l;
			break;
		}
    }

    // free memory
	convolution_delete(presmoothing);
	free(presmooth_filter);

	// wx and wy in each layer
	image_t *wx_layer = wx, *wy_layer = wy;

	// new object if more than one layer and use rescaled wx,wy as initialization
	if(L > 1) {//
		// get scaling factor
		float fx = (1.0f*pyramid[L - 1][0]->width) / pyramid[0][0]->width;		//pow(p_scale,(L - 1));
		float fy = (1.0f*pyramid[L - 1][0]->height) / pyramid[0][0]->height;	//pow(p_scale,(L - 1));

		Mat tmpx(pyramid[0][0]->height, pyramid[0][0]->width, CV_32FC1);
		Mat tmpy(pyramid[0][0]->height, pyramid[0][0]->width, CV_32FC1);
		img2Mat(wx, &tmpx);
		img2Mat(wy, &tmpy);

		resize(tmpx, tmpx, Size(pyramid[L - 1][0]->width, pyramid[L - 1][0]->height), 0, 0, INTER_LINEAR);
		resize(tmpy, tmpy, Size(pyramid[L - 1][0]->width, pyramid[L - 1][0]->height), 0, 0, INTER_LINEAR);

		wx_layer = image_new(tmpx.cols, tmpx.rows);
		wy_layer = image_new(tmpy.cols, tmpy.rows);
		mat2Img(&tmpx, wx_layer);
		mat2Img(&tmpy, wy_layer);
		image_mul_scalar(wx_layer, fx);							// scale flow vectors
		image_mul_scalar(wy_layer, fy);							// scale flow vectors
	}

	// iterate over each layer
    for(int l = L - 1; l >= 0; l--) {
    	// print layer
    	if(L > 1) cout << "layer " << l << ":" << endl;

    	// upscale flow field in each layer besides first layer
    	if(l < (L - 1)) {
			// rescale flow field and vectors
			image_t *tmp_x, *tmp_y ;

			// use temporary flow fields for rescaled layers and wx,wy for last layer
			if(l > 0) {
				tmp_x = image_new(pyramid[l][0]->width, pyramid[l][0]->height);
				tmp_y = image_new(pyramid[l][0]->width, pyramid[l][0]->height);
			} else {
				tmp_x = wx;
				tmp_y = wy;
			}

			// get scaling factor
			float fx = (1.0f*pyramid[l][0]->width) / pyramid[l+1][0]->width;
			float fy = (1.0f*pyramid[l][0]->height) / pyramid[l+1][0]->height;

			Mat tmpx(wx_layer->height, wx_layer->width, CV_32FC1);
			Mat tmpy(wy_layer->height, wy_layer->width, CV_32FC1);
			img2Mat(wx_layer, &tmpx);
			img2Mat(wy_layer, &tmpy);

			resize(tmpx, tmpx, Size(pyramid[l][0]->width, pyramid[l][0]->height), 0, 0, INTER_LINEAR);
			resize(tmpy, tmpy, Size(pyramid[l][0]->width, pyramid[l][0]->height), 0, 0, INTER_LINEAR);

			mat2Img(&tmpx, tmp_x);
			mat2Img(&tmpy, tmp_y);
			image_mul_scalar(tmp_x, fx);									// scale flow vectors
			image_mul_scalar(tmp_y, fy);									// scale flow vectors

			image_delete(wx_layer);
			image_delete(wy_layer);
			wx_layer = tmp_x;
			wy_layer = tmp_y;
    	}

    	// output flow field of each layer if verbose > 1
    	if(params.verbosity(VER_FLO_PYR)) {
			Mat tmp_mat = Mat(wx_layer->height, wx_layer->width, CV_32FC1);
			double min, max;

			img2Mat(wx_layer, &tmp_mat);
			cv::minMaxLoc(tmp_mat, &min, &max);
			tmp_mat = (tmp_mat - min) / (max  - min);
			tmp_mat.convertTo(tmp_mat, CV_8UC1, 255);

			cv::minMaxLoc(tmp_mat, &min, &max);

			stringstream title;
			title << "Flow u " << l;
			namedWindow( title.str(), WINDOW_FREERATIO );               // Create a window for display.
			imshow( title.str(), tmp_mat );                            // Show our image inside it.4
			waitKey(0);

			tmp_mat = Mat(wx_layer->height, wx_layer->width, CV_32FC1);
			img2Mat(wy_layer, &tmp_mat);
			cv::minMaxLoc(tmp_mat, &min, &max);
			tmp_mat = (tmp_mat - min) / (max  - min);
			tmp_mat.convertTo(tmp_mat, CV_8UC1, 255);

			cv::minMaxLoc(tmp_mat, &min, &max);

			title << "Flow v " << l;
			namedWindow( title.str(), WINDOW_FREERATIO );               // Create a window for display.
			imshow( title.str(), tmp_mat );                            // Show our image inside it.
			waitKey(0);
		}

		// compute one layer
    	if(l == 0) params.setParameter<int>("final", 1);
    	else params.setParameter<int>("final", 0);

    	avg_change = compute_one_level(var_aux, wx_layer, wy_layer, pyramid[l], params);
    }

    params.setParameter<int>("final", 0);

	// free memory
    convolution_delete(deriv);
    convolution_delete(deriv_flow);
    deriv = NULL;
    deriv_flow = NULL;

    for(int l = 0; l < L; l++) {
		for(int s = 0; s < F; s++)
			color_image_delete(pyramid[l][s]);

		delete[] pyramid[l];
    }
	delete[] pyramid;

	if(channel_w == NULL)
		color_image_delete(var_aux.channel_w);

	return avg_change;
}



void Variational_MT::mat2colorImg(Mat *src, color_image_t *dst) {
	int ch = src->channels();
	for(int i=0 ; i < dst->height ; i++){
		for(int j=0 ; j < dst->width; j++) {
			if(ch == 1) {
				dst->c1[i*dst->stride+j] = src->at<float>(i,j); // they use 8 bit images
				dst->c2[i*dst->stride+j] = src->at<float>(i,j);
				dst->c3[i*dst->stride+j] = src->at<float>(i,j);
			} else {
				dst->c1[i*dst->stride+j] = src->at<Vec3f>(i,j)[0]; // they use 8 bit images
				dst->c2[i*dst->stride+j] = src->at<Vec3f>(i,j)[1];
				dst->c3[i*dst->stride+j] = src->at<Vec3f>(i,j)[2];
			}
		}
	}
}

void Variational_MT::colorImg2Mat(color_image_t *src, Mat *dst) {
	int ch = dst->channels();
	for(int i=0 ; i < dst->rows ; i++){
		for(int j=0 ; j < dst->cols; j++) {
			if(ch == 1) {
				dst->at<float>(i,j) = (src->c1[i*src->stride+j] + src->c2[i*src->stride+j] + src->c3[i*src->stride+j]) / 3;
			} else {
				dst->at<Vec3f>(i,j)[0] = src->c1[i*src->stride+j]; // they use 8 bit images
				dst->at<Vec3f>(i,j)[1] = src->c2[i*src->stride+j];
				dst->at<Vec3f>(i,j)[2] = src->c3[i*src->stride+j];
			}
		}
	}
}

void Variational_MT::img2Mat(const image_t *src, Mat *dst) {
	int ch = dst->channels();
	for(int i=0 ; i < dst->rows; i++){
		for(int j=0 ; j < dst->cols; j++) {
			if(ch == 1) {
				dst->at<float>(i,j) = src->data[i*src->stride+j] ;
			} else {
				dst->at<Vec3f>(i,j)[0] = src->data[i*src->stride+j]; // they use 8 bit images
				dst->at<Vec3f>(i,j)[1] = src->data[i*src->stride+j];
				dst->at<Vec3f>(i,j)[2] = src->data[i*src->stride+j];
			}
		}
	}
}



void Variational_MT::mat2Img(const Mat* img, image_t *seq) {
	for(int i=0 ; i<seq->height ; i++){
		for( int j=0 ; j<seq->width ; j++){
			seq->data[i*seq->stride+j] = img->at<float>(i,j); // they use 8 bit images
		}
	}
}
