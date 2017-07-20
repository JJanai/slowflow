/*
 * dense_tracking.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: Janai
 */
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <omp.h>
#include <random>
#include <set>

#include <flann/flann.hpp>

extern "C" {
	#include "../libs/dmgunturk/dmha.h"
}
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/indexed_by.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/chrono/thread_clock.hpp>

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_randist.h>

#include "epic_flow_extended/image.h"
#include "epic_flow_extended/io.h"
#include "epic_flow_extended/epic.h"
#include "epic_flow_extended/variational_mt.h"
#include "utils/utils.h"
#include "utils/hypothesis.h"
#include "utils/parameter_list.h"
#include "penalty_functions/penalty_function.h"
#include "penalty_functions/lorentzian.h"
#include "penalty_functions/modified_l1_norm.h"
#include "penalty_functions/quadratic_function.h"
#include "configuration.h"


using namespace std;
using namespace cv;
using namespace boost::multi_index;
using namespace boost::chrono;

void setDefaultVariational(ParameterList& params) {
    // general
    params.insert("verbose", "0", true);
    params.insert("threads", "1", true);

    params.insert("scale", "1.0f", true);

    params.insert("slow_flow_S", "2", true);

    // energy function
    params.insert("slow_flow_alpha", "4.0f");
    params.insert("slow_flow_gamma", "6.0f", true);
    params.insert("slow_flow_delta", "1.0f", true);

    // image pyramid
    params.insert("slow_flow_layers", "1", true);
    params.insert("slow_flow_p_scale", "0.9f", true);

    // optimization
    params.insert("slow_flow_niter_alter", "10", true);
    params.insert("slow_flow_niter_outer", "10", true);
    params.insert("slow_flow_thres_outer", "1e-5", true);
    params.insert("slow_flow_niter_inner", "1", true);
    params.insert("slow_flow_thres_inner", "1e-5", true);
    params.insert("slow_flow_niter_solver", "30", true);
    params.insert("slow_flow_sor_omega", "1.9f", true);

    // occlusion reasoning
    params.insert("slow_flow_occlusion_reasoning", "1", true);
    params.insert("slow_flow_occlusion_penalty", "0.1", true);
    params.insert("slow_flow_occlusion_alpha", "0.1", true);
    params.insert("slow_flow_output_occlusions", "1", true);

    // regularization
    params.insert("slow_flow_robust_color", "1", true);
    params.insert("slow_flow_robust_color_eps", "0.001", true);
    params.insert("slow_flow_robust_color_truncation", "0.5", true);
    params.insert("slow_flow_robust_reg", "1", true);
    params.insert("slow_flow_robust_reg_eps", "0.001", true);
    params.insert("slow_flow_robust_reg_truncation", "0.5", true);
}

void setDefault(ParameterList& params) {
    params.insert("verbose", "0", true);
	// dense tracking parameters
	params.insert("scale", "1");						// scale input images
	params.insert("acc_skip_pixel", "1");				// skip number of pixel for final resolution)
	params.insert("acc_occlusion", "0");				// set to 1 to use occlusions from jet optimization

	// data driven proposal generation
	params.insert("acc_consistency_threshold", "1.0");  // threshold for forward backward consistency
	params.insert("acc_discard_inconsistent", "1");		// set to 1 to discard inconsistent trajectories
	params.insert("acc_epic_interpolation", "1");		// set to 1 to use epic flow interpolation
	params.insert("acc_epic_skip", "2");				// skip pixel for epic flow interpolation

	// energy weights and penalites
	params.insert("acc_jet_consistency", "1.0");		// weight of flow data term
	params.insert("acc_brightness_constancy", "0.1");	// weight of brightness constancy
	params.insert("acc_gradient_constancy", "1.0");		// weight of gradient constancy
	params.insert("acc_occlusion_penalty", "500.0");	// penalty for occluded frames
	params.insert("acc_beta", "10.0");					// spatial smoothness flow
	params.insert("acc_satial_occ", "10.0");			// spatial smoothness occlusion
	params.insert("acc_temporal_occ", "10.0");			// temporal smoothness occlusion
	params.insert("acc_cv", "0.0");						// temporal flow smoothness

	params.insert("acc_traj_sim_method", "1");			// spatial trajectory smoothness compares 0: adjacent flow, 1: accumulated flow, 2: final flow
	params.insert("acc_traj_sim_thres", "0.1");			// threshold for distance between two hypotheses

	// occlusion initialization
	params.insert("acc_occlusion_threshold", "5.0");	// threshold for comparison of high speed flow and trajectory
	params.insert("acc_occlusion_fb_threshold", "5.0");	// threshold for forward backward consistency check

	// discrete optimization
	params.insert("acc_alternate", "5");				// alternate between trws and propagation
	params.insert("acc_approach", "0");					// 0: TRW-S, 1: BP
	params.insert("acc_trws_eps", "1e-5");				// stopping criterion for lower bound
	params.insert("acc_trws_max_iter", "10");			// maximum number of iterations

	// neighbor propagation
	params.insert("acc_neigh_hyp", "5");				// number of proposals propagated to neighbors
	params.insert("acc_neigh_hyp_radius", "100.0");		// radius for proposal propagation
	params.insert("acc_neigh_skip1", "2");				// skipped pixel for nearest neighbor search tree
	params.insert("acc_neigh_skip2", "4");				// skipped pixel for nearest neighbor search tree
	params.insert("acc_hyp_neigh_tryouts", "20");		// maximum number of tryouts for sampling

	// penalty
	params.insert("acc_penalty_fct_data", "1");			// penality fct data term 0: quadratic 1: modified l1 norm 2: lorentzian
	params.insert("acc_penalty_fct_data_eps", "0.001");
	params.insert("acc_penalty_fct_reg", "1");			// penality fct regularization term 0: quadratic 1: modified l1 norm 2: lorentzian
	params.insert("acc_penalty_fct_reg_eps", "0.001");
}

inline bool insideImg(double x, double y, int width, int height) {
	return (y >= 0 && y < height && x >= 0 && x < width);
}

bool compareHypotheses(hypothesis* h1, hypothesis* h2) {
	return (h1 != NULL && (h2 == NULL || h1->score() < h2->score()));	// ascending order considering the score and null hypothesis to the end
}

float addJC(hypothesis* h, const Mat* obs, double acc_jc, double acc_cv, PenaltyFunction* phi_d, ParameterList& params, const Mat* occlusion_masks = NULL) {
	Point2d p = h->p;

	int height = obs[0].rows;
	int width = obs[0].cols;

	double jenergy = 0;
	double cvenergy = 0;
	int contribution = 0;

	for (uint32_t j = 0; j < params.Jets; j++) {
		double u_j = h->u(j);
		double v_j = h->v(j);  // x, y

		double u_jm1 = 0;
		double v_jm1 = 0;
		if (j > 0) {
			u_jm1 = h->u((j - 1));
			v_jm1 = h->v((j - 1));  // x, y
		}

		// exclude unknown flow
		if (u_j > UNKNOWN_FLOW_THRESH || v_j > UNKNOWN_FLOW_THRESH)
			break;

		// ############################################### compare flow to jet estimation
		if (insideImg(p.x + u_jm1, p.y + v_jm1, width, height)) {
			if (h->occluded(j) == 1 || h->occluded(j+1) == 1)	// flow from j to j+1
				continue;

			double I_x = bilinearInterp<double>(p.x + u_jm1, p.y + v_jm1, obs[j], 1);
			double I_y = bilinearInterp<double>(p.x + u_jm1, p.y + v_jm1, obs[j], 0);  // x, y

			jenergy += 0.5 * phi_d->apply(((u_j - u_jm1 - I_x) * (u_j - u_jm1 - I_x) + (v_j - v_jm1 - I_y) * (v_j - v_jm1 - I_y)));

			contribution++;
		}

		double u_jp1 = 0;
		double v_jp1 = 0;
		if (j + 1 < params.Jets) {
			u_jp1 = h->u((j + 1));
			v_jp1 = h->v((j + 1));  // x, y
		}

		// ############################################### assume constant velocity
		double u_sq = 2 * u_j - u_jm1 - u_jp1;
		double v_sq = 2 * v_j - v_jm1 - v_jp1;
		u_sq *= u_sq;
		v_sq *= v_sq;

		cvenergy += sqrt(u_sq + v_sq);
	}

	if(contribution > 0) jenergy /= contribution;

	return acc_jc * jenergy + acc_cv * cvenergy;
}

double dt_warp_time = 0, dt_med_time = 0, dt_sum_time = 0;

/*
 *  ################################### CONSIDER ALL POSSIBLE PAIRWISE TERMS #########################################
 */
float addBCGC(hypothesis* h, color_image_t const* const * obs, color_image_t const* const * dx, color_image_t const* const * dy, double acc_bc, double acc_gc, int skip, ParameterList& params, const Mat* occlusion_masks = NULL) {
	Point2d p = h->p;

	int r = 0.5f * (skip + 1);

	int height = obs[0]->height;
	int width = obs[0]->width;
	int stride = obs[0]->stride;

	double wenergy = 0;
	double neighs = 0;

	time_t dt_start, dt_end;

	// ############################# Mean brightness and gradient constancy #############################
	for (int off_x = (p.x - r); off_x <= (p.x + r); off_x++) {
		for (int off_y = (p.y - r); off_y <= (p.y + r); off_y++) {
			if (off_x < 0 || off_x >= width || off_y < 0 || off_y >= height)
				continue;

			// warp images
			uint32_t visible = 0;
			vector<vector<double> > I(3, vector<double>(params.Jets + 1, 0));
			vector<vector<double> > Ix(3, vector<double>(params.Jets + 1, 0));
			vector<vector<double> > Iy(3, vector<double>(params.Jets + 1, 0));

			time(&dt_start);
			for (uint32_t j = 0; j < params.Jets + 1; j++) {
				double x_j = off_x;
				double y_j = off_y;

				if (j == 0) {
					int idx = stride * y_j + x_j;
					I[0][j] = obs[j]->c3[idx];
					I[1][j] = obs[j]->c2[idx];
					I[2][j] = obs[j]->c1[idx];
					Ix[0][j] = dx[j]->c3[idx];
					Ix[1][j] = dx[j]->c2[idx];
					Ix[2][j] = dx[j]->c1[idx];
					Iy[0][j] = dy[j]->c3[idx];
					Iy[1][j] = dy[j]->c2[idx];
					Iy[2][j] = dy[j]->c1[idx];

					visible++;
				} else {
					x_j += h->u(j - 1);
					y_j += h->v(j - 1);

					// stop in the case the object will stay occluded!
					if (insideImg(x_j, y_j, width, height) && (occlusion_masks == NULL || occlusion_masks[j].at<uchar>(y_j, x_j) != 0)) {
						I[0][j] = bilinearInterp(x_j, y_j, obs[j]->c3, height, width, stride);
						I[1][j] = bilinearInterp(x_j, y_j, obs[j]->c2, height, width, stride);
						I[2][j] = bilinearInterp(x_j, y_j, obs[j]->c1, height, width, stride);
						Ix[0][j] = bilinearInterp(x_j, y_j, dx[j]->c3, height, width, stride);
						Ix[1][j] = bilinearInterp(x_j, y_j, dx[j]->c2, height, width, stride);
						Ix[2][j] = bilinearInterp(x_j, y_j, dx[j]->c1, height, width, stride);
						Iy[0][j] = bilinearInterp(x_j, y_j, dy[j]->c3, height, width, stride);
						Iy[1][j] = bilinearInterp(x_j, y_j, dy[j]->c2, height, width, stride);
						Iy[2][j] = bilinearInterp(x_j, y_j, dy[j]->c1, height, width, stride);

						visible++;
					}
				}
			}
			time(&dt_end);
			dt_warp_time += difftime(dt_end, dt_start);

			int contribution = 0;
			double e_p = 0;


			time(&dt_start);
			// compute data terms
			for (uint32_t i = 0; i < visible; i++) {
				for (uint32_t j = (i + 1); j < visible; j++) {
					double x_i = off_x;
					double y_i = off_y;
					if (i > 0) {
						x_i += h->u(i - 1);
						y_i += h->v(i - 1);
					}
					double x_j = off_x + h->u(j - 1);
					double y_j = off_y + h->v(j - 1);  // x, y

					if (insideImg(x_i, y_i, width, height) && insideImg(x_j, y_j, width, height)) {
						if (h->occluded(i) == 1 || h->occluded(j) == 1)
							continue;															// skip occluded jets

						e_p += acc_bc * 0.3334  * (fabs(I[0][i] - I[0][j]) + fabs(I[1][i] - I[1][j]) + fabs(I[2][i] - I[2][j]));
						e_p += acc_gc * 0.3334 * (fabs(Ix[0][i] - Ix[0][j]) + fabs(Ix[1][i] - Ix[1][j]) + fabs(Ix[2][i] - Ix[2][j]) + fabs(Iy[0][i] - Iy[0][j]) + fabs(Iy[1][i] - Iy[1][j]) + fabs(Iy[2][i] - Iy[2][j]));

						contribution++;
					}
				}
			}
			time(&dt_end);
			dt_sum_time += difftime(dt_end, dt_start);


			if(contribution > 0) e_p /= contribution;

			wenergy += e_p;
			neighs++;
		}
	}

	if(neighs > 0) wenergy /= neighs;

	return wenergy;
}

float addOC(hypothesis* h, double acc_occ, double acc_temporal_occ, ParameterList& params) {
	int occlusions = 0;
	int change = 0;

	for (uint32_t i = 0; i < params.Jets + 1; i++) {
		// ############################# prefer smaller number of occlusions #############################
		occlusions += h->occluded(i);															// skip occluded jets

		// ############################# expect temporal consistancy #############################
		if (i < params.Jets && h->occluded(i) != h->occluded(i + 1))
			change++;
	}

	return acc_occ * occlusions + acc_temporal_occ * change;
}

void computeSmoothnessWeight(const color_image_t *im, image_t* lum, float coef, const convolution_t *deriv, float avg_1, float avg_2, float avg_3, float std_1, float std_2, float std_3, bool hbit) {
	int i;
	image_t *lum_x = image_new(im->width, im->height), *lum_y = image_new(im->width, im->height);

	// compute luminance
	v4sf *im1p = (v4sf*) im->c1, *im2p = (v4sf*) im->c2, *im3p = (v4sf*) im->c3, *lump = (v4sf*) lum->data;
	for (i = 0; i < im->height * im->stride / 4; i++) {
		if (hbit)
			*lump = (0.299f * ((*im1p) * std_1 + avg_1) + 0.587f * ((*im2p) * std_2 + avg_2) + 0.114f * ((*im3p) * std_3 + avg_3)) / 65535.0f;	// channels normalized to 0..1
		else
			*lump = (0.299f * ((*im1p) * std_1 + avg_1) + 0.587f * ((*im2p) * std_2 + avg_2) + 0.114f * ((*im3p) * std_3 + avg_3)) / 255.0f;	// channels normalized to 0..1

		lump += 1;
		im1p += 1;
		im2p += 1;
		im3p += 1;
	}

	// compute derivatives with five-point tencil
	convolve_horiz(lum_x, lum, deriv);
	convolve_vert(lum_y, lum, deriv);

	// compute lum norm
	lump = (v4sf*) lum->data;
	v4sf *lumxp = (v4sf*) lum_x->data, *lumyp = (v4sf*) lum_y->data;
	for (i = 0; i < lum->height * lum->stride / 4; i++) {
		*lump = -coef * __builtin_ia32_sqrtps((*lumxp) * (*lumxp) + (*lumyp) * (*lumyp));
		lump[0][0] = 0.5f * expf((float) lump[0][0]);
		lump[0][1] = 0.5f * expf((float) lump[0][1]);
		lump[0][2] = 0.5f * expf((float) lump[0][2]);
		lump[0][3] = 0.5f * expf((float) lump[0][3]);

		lump += 1;
		lumxp += 1;
		lumyp += 1;
	}

	image_delete(lum_x);
	image_delete(lum_y);
}

/* show usage information */
void usage(){
    printf("usage:\n");
    printf("    ./dense_tracking [cfg] -select [estimation for one specific final pair] -resume\n");
    printf("\n");
}

int main(int argc, char **argv) {
	if (argc < 2) {
		usage();
		return -1;
	}

	/*
	 *  getting config file location
	 */
	uint32_t selected = 0;
	uint32_t selected_end = 0;

	/*
	 *  read in parameters and print config
	 */
	string file = string(argv[1]);

	if (boost::filesystem::exists(file)) {
		printf("using parameters %s\n", file.c_str());
	} else {
		usage();
		return -1;
	}

	ParameterList params;
	setDefault(params);
	params.read(file);

    bool resume_frame = false;
	int max_fps = params.parameter<int>("max_fps", "0");		// fps of original sequence
	int threads = params.parameter<int>("threads", "1");

	#define isarg(key)  !strcmp(a,key)
	if (argc > 1) {
		int current_arg = 1;
		while (current_arg < argc) {
			const char* a = argv[current_arg++];
			if (a[0] != '-') {
				continue;
			}

			if ( isarg("-h") || isarg("-help"))
				usage();
			else if (isarg("-output"))
				params.output = string(argv[current_arg++]);
			else if (isarg("-threads"))
				threads = atoi(argv[current_arg++]);
			else if( isarg("-resume") )
				resume_frame = true;
			else if( isarg("-select") ) {
				selected = atoi(argv[current_arg++]);
				selected_end = selected + 1;
			} else {
				fprintf(stderr, "unknown argument %s", a);
				usage();
				exit(1);
			}
		}
	} else {
		usage();
		return -1;
	}

	// check path
	for (uint32_t i = 0; i < params.jet_estimation.size(); i++)
		if (params.jet_estimation[i].back() != '/') params.jet_estimation[i] += "/";


	bool sintel = params.parameter<bool>("sintel", "0");
	bool subframes = params.parameter<bool>("subframes", "0");	// are subframes specified

	int skip_pixel = params.parameter<int>("acc_skip_pixel", "0");

	int ref_fps_F = params.parameter<int>("ref_fps_F", "1");
	uint32_t rates = params.jet_estimation.size();
	vector<float> weight_jet_estimation(rates, 0);
	for (uint32_t i = 0; i < rates; i++) {
		weight_jet_estimation[i] = i;
		if(params.jet_weight.size() > i)
			weight_jet_estimation[i] = params.jet_weight[i];
	}
	int min_fps_idx = params.parameter<int>("acc_min_fps", "0");

	/*
	 * ###############################  get step sizes ###############################
	 */
	if (rates != params.jet_S.size()) {
		bool error = false;
		params.jet_S.resize(rates);
		for(uint32_t r = 0; r < rates; r++) {
			if(access((params.jet_estimation[r] + "/config.cfg").c_str(), F_OK) != -1) {
				ParameterList tmp((params.jet_estimation[r] + "/config.cfg"));

				if(tmp.exists("slow_flow_S"))
					params.jet_S[r] = tmp.parameter<int>("slow_flow_S");
				else {
					cerr << "Error reading S from " << params.jet_estimation[r] << "/config.cfg" << endl;
					error = true;
				}

			} else {
				cerr << "Error reading " << params.jet_estimation[r] << "/config.cfg" << endl;
				error = true;
			}
		}

		if(error) {
			cerr << "Frame rate or window size for Jet estimations missing!" << endl;
			exit(-1);
		}
	}

	// first slow flow estimation is used as reference
	int steps = params.jet_S[min_fps_idx] - 1;

	/*
	 *  get frame rates
	 */
	if (rates <= 0) {
		cerr << "No Jet estimation specified!" << endl;
		exit(-1);
	}
	if (rates != params.jet_fps.size()) {
		bool error = false;
		params.jet_fps.resize(rates);

		for(uint32_t r = 0; r < rates; r++) {
			if(access((params.jet_estimation[r] + "/config.cfg").c_str(), F_OK) != -1) {
				ParameterList tmp((params.jet_estimation[r] + "/config.cfg"));

				if(tmp.exists("jet_fps")) {
					params.jet_fps[r] = tmp.parameter<int>("jet_fps");
				} else {
					cerr << "Error reading jet_fps from " << params.jet_estimation[r] << "/config.cfg" << endl;
					error = true;
				}
			} else {
				cerr << "Error reading " << params.jet_estimation[r] << "/config.cfg" << endl;
				error = true;
			}
		}

		if(error) {
			cerr << "Frame rate or window size for Jet estimations missing!" << endl;
			exit(-1);
		}
	}

	// ############################### specify reference framerate by maximum flow ########################################################
	params.Jets = params.jet_fps[min_fps_idx] / (1.0f * params.parameter<int>("ref_fps") * steps);

	/*
	 * decompose sequence in subsets and adjust number of frames if necessary
	 */
	uint32_t Jets = params.Jets;

	int skip = (1.0f * max_fps) / params.jet_fps[min_fps_idx];													// number of frames to skip for jet fps

	uint32_t gtFrames = params.Jets * skip;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	// MAKE SURE FOLDER IS NOT OVERWRITTEN
	if(!resume_frame) {
		string newPath = params.output;
		if(newPath[newPath.length() - 1] == '/') newPath.erase(newPath.length() - 1);
		int num = 1;
		while(boost::filesystem::exists(newPath)) {
			cerr << newPath << " already exists!" << endl;
			stringstream tmp;
			tmp << newPath << "_" << num++;
			newPath = tmp.str();
		}
		params.output = newPath;
	}
	if(params.output[params.output.length() - 1] != '/') params.output = params.output + "/";

	boost::filesystem::create_directories(params.output);               				// accumulate folder

	ofstream infos;
	infos.open((params.output + "/config.cfg").c_str());
	infos << "# Slow Flow Accumulation\n";
	infos << params.cfgString(true);
	infos.close();
	cout << (params.output + "config.cfg").c_str() << endl;

	double traj_sim_thres = params.parameter<double>("acc_traj_sim_thres");
	int traj_sim_method = params.parameter<int>("acc_traj_sim_method");

	bool use_oracle = params.parameter<bool>("acc_oracle");
	bool use_occlusion = params.parameter<bool>("acc_occlusion");
	float acc_jc = params.parameter<float>("acc_jet_consistency");
	float acc_bc = params.parameter<float>("acc_brightness_constancy");
	float acc_gc = params.parameter<float>("acc_gradient_constancy");
	float acc_occ = params.parameter<float>("acc_occlusion_penalty");

	double acc_beta = params.parameter<double>("acc_beta");										// weight for flow spatial smoothness
	double acc_spatial_occ = params.parameter<double>("acc_spatial_occ");								// weight for occlusion spatial smoothness
	double acc_temporal_occ = params.parameter<double>("acc_temporal_occ");								// weight for occlusion spatial smoothness
	double acc_cv = params.parameter<double>("acc_cv");											// weight for constant velocity
	double outlier_beta = params.parameter<double>("acc_outlier_beta");							// energy for occlusion penalty

	float occlusion_threshold = params.parameter<float>("acc_occlusion_threshold");							// hypotheses from neighbors
	float occlusion_fb_threshold = params.parameter<float>("acc_occlusion_fb_threshold");							// hypotheses from neighbors

	// propagate hypotheses to neighbors
	int alternate = params.parameter<int>("acc_alternate");									// alternate between trws and perturbation
	uint32_t perturb_keep = params.parameter<int>("acc_perturb_keep");							// top x hypotheses will be kept
	bool discard_inconsistent = params.parameter<bool>("acc_discard_inconsistent");
	bool use_jet_occlusions = params.parameter<bool>("acc_use_jet_occlusions");

	uint32_t hyp_neigh = params.parameter<int>("acc_neigh_hyp");									// hypotheses from neighbors
	float hyp_neigh_radius = params.parameter<int>("acc_neigh_hyp_radius");					// radius to draw neighbors from
	bool draw_nn_from_radius = (hyp_neigh_radius > 0);
	float hyp_neigh_draws = params.parameter<int>("acc_neigh_draws");					// radius to draw neighbors from
	int hyp_neigh_tryouts = params.parameter<int>("acc_hyp_neigh_tryouts");					// maximum number of tryouts for sampling

	int nn_skip1 = params.parameter<int>("acc_neigh_skip1", "2");
	int nn_skip2 = params.parameter<int>("acc_neigh_skip2", "4");

	// discrete optimization robust penalty function
	int penalty_fct_data = params.parameter<int>("acc_penalty_fct_data");						// 0:QUADRATIC, 1:MOD_L1, 2:LORETZIAN
	double penalty_fct_data_eps = params.parameter<double>("acc_penalty_fct_data_eps");
	int penalty_fct_reg = params.parameter<int>("acc_penalty_fct_reg");							// 0:QUADRATIC, 1:MOD_L1, 2:LORETZIAN
	double penalty_fct_reg_eps = params.parameter<double>("acc_penalty_fct_reg_eps");

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	if (params.exists("seed"))
		seed = params.parameter<int>("seed");
	default_random_engine generator(seed);

	// EPIC FLOW
	float epic_skip = params.parameter<float>("acc_epic_skip", "2");
	epic_params_t epic_params;
	epic_params_default(&epic_params);
	epic_params.pref_nn = 25;
	epic_params.nn = 160;
	epic_params.coef_kernel = 1.1f;

	// discrete optimization set options
	MRFEnergy<TypeGeneral>::Options options;
	options.m_method = params.parameter<int>("acc_approach");  							// 0: TRW-S, 1: BP
	options.m_eps = params.parameter<double>("acc_trws_eps");
	options.m_iterMax = params.parameter<int>("acc_trws_max_iter");
	options.m_verbosityLevel = params.verbosity(VER_CMD);
	options.m_printIter = 1;
	options.m_printMinIter = 0;

	// nearest neighbor search
	flann::SearchParams flann_params;
	flann_params.checks = 128;				//
	flann_params.max_neighbors = -1;		// unlimited
	flann_params.sorted = true;

	flann::SearchParams flann_params_p;
	flann_params_p.checks = 128;				//
	flann_params_p.max_neighbors = -1;		// unlimited
	flann_params_p.sorted = true;

	// discrete optimization set robust function
	PenaltyFunction *phi_d = NULL, *phi_s = NULL;
	switch (penalty_fct_data) {
		case 0:
			phi_d = new QuadraticFunction();
			break;
		case 1:
			phi_d = new ModifiedL1Norm(penalty_fct_data_eps);
			break;
		default:
			phi_d = new Lorentzian(penalty_fct_data_eps);
			break;
	}
	switch (penalty_fct_reg) {
		case 0:
			phi_s = new QuadraticFunction();
			break;
		case 1:
			phi_s = new ModifiedL1Norm(penalty_fct_reg_eps);
			break;
		default:
			phi_s = new Lorentzian(penalty_fct_reg_eps);
			break;
	}

	stringstream acc_folder;

	acc_folder << params.output;

	if(use_oracle)
		cout << endl << "Oracle uses same penalty as first jet estimation!!!!" << endl;

	boost::filesystem::create_directories(acc_folder.str());
	boost::filesystem::create_directories(acc_folder.str() + "gt_occlusions/"); // accumulate folder
	boost::filesystem::create_directories(acc_folder.str() + "occlusions/");  	// accumulate folder
	boost::filesystem::create_directories(acc_folder.str() + "tmp/");           // accumulate folder

	params.insert("accumulation", acc_folder.str() + "frame_%i.flo", true);

	params.print();

	// in case of sintel data we would like to be able to distinguish frame number from 24 fps and 1008 fps
	if (sintel && !subframes)
		params.sequence_start = params.sequence_start * 1000; 	//


	if(selected_end == 0)
		selected_end = ref_fps_F;

	// iterate while changing start
	#pragma omp parallel for num_threads(threads) schedule(static,1)
	for(uint32_t start_jet = selected; start_jet < selected_end; start_jet++) {

		// alternate between continuous and discrete optimization
		stringstream numVariablesStream, factorsStream;
		double avg_unary_time = 0, avg_pairw_time = 0, avg_optimization_time = 0;

		// thread safe
		ParameterList thread_params;
		#pragma omp critical
		{
			thread_params = ParameterList(params);
		}

		// update start jet
		thread_params.sequence_start = thread_params.sequence_start + start_jet * Jets * steps * skip;

		int start_format = (thread_params.file.find_last_of('/') + 1);
		int end_format = thread_params.file.length() - start_format;

		string sequence_path = thread_params.file.substr(0, start_format);

		string format = "frame_%i.png";
		string flow_format = thread_params.parameter<string>("flow_format", "frame_%i");

		int len_format = flow_format.find_last_of('.');
		flow_format = flow_format.substr(0, len_format);
		format = thread_params.file.substr(start_format, end_format);
		if (sequence_path[sequence_path.length() - 1] != '/')
			sequence_path = sequence_path + "/";


		char final_file[1024];
		if (!sintel)
			sprintf(final_file, (acc_folder.str() + "/" + flow_format + ".flo").c_str(), thread_params.sequence_start);
		else
			sprintf(final_file, (acc_folder.str() + "/s" + flow_format + ".flo").c_str(), thread_params.sequence_start, 0);

		if (access(final_file, F_OK) != -1) {
			cout << "Flow file " << final_file << " already exists!" << endl;
			continue;
		}
		cout << "Flow file " << final_file << " does not exists!" << endl;

		/*
		 * ################### read in image sequence ###################
		 */
		vector<int> red_loc = thread_params.splitParameter<int>("raw_red_loc", "0,0");

		// read in sequence, forward and backward flow, and segmentation
		Mat *sequence = new Mat[Jets + 1];  						// high speed video sequence
		Mat reference;
		color_image_t **data = new color_image_t*[Jets + 1];  		// high speed video sequence
		color_image_t **data_dx = new color_image_t*[Jets + 1];  	// derivative of high speed video sequence
		color_image_t **data_dy = new color_image_t*[Jets + 1];  	// derivative of high speed video sequence
		Mat* gt = NULL;
		image_t*** gt_sfr = NULL;
		Mat* gt_occlusions = NULL;
		Mat* occlusions = new Mat[Jets];

		Mat *forward_flow = new Mat[Jets];
		Mat *backward_flow = new Mat[Jets];
		/*
		 * ################### read in image sequence ###################
		 */
		float norm = 1;
		for (uint32_t f = 0; f < (Jets + 1); f++) {
			char img_file[1024];
			if (!sintel) {
				sprintf(img_file, (sequence_path + format).c_str(), thread_params.sequence_start + f * steps * skip);
			} else {
				int sintel_frame = thread_params.sequence_start / 1000;
				int hfr_frame = f * steps * skip + (thread_params.sequence_start % 1000);

				while (hfr_frame < 0) {
					sintel_frame--;
					hfr_frame = 42 + hfr_frame;
				}
				while (hfr_frame > 41) {
					sintel_frame++;
					hfr_frame = hfr_frame - 42;
				}

				sprintf(img_file, (sequence_path + format).c_str(), sintel_frame, hfr_frame);
			}
			cout << "Reading " << img_file << "..." << endl;

			sequence[f] = imread(string(img_file), CV_LOAD_IMAGE_UNCHANGED);         // load images

			if (sequence[f].type() == 2 || sequence[f].type() == 18)
				norm = 1.0f / 255;		// for 16 bit images

			// convert to floating point
			sequence[f].convertTo(sequence[f], CV_32FC(sequence[f].channels()));

			/*
			 * DEMOSAICING
			 */
			if (thread_params.exists("raw") && thread_params.parameter<bool>("raw")) {
				Mat tmp = sequence[f].clone();
				color_image_t* tmp_in = color_image_new(sequence[f].cols, sequence[f].rows);
				color_image_t* tmp_out = color_image_new(sequence[f].cols, sequence[f].rows);

				switch (thread_params.parameter<int>("raw_demosaicing", "0")) {
					case 0: // use bilinear demosaicing
						sequence[f] = Mat::zeros(tmp.rows, tmp.cols, CV_32FC3);
						//						bayer2rgb(tmp, sequence[f], green_start, blue_start); 	// red green
						bayer2rgbGR(tmp, sequence[f], red_loc[0], red_loc[1]); // red green
						break;

					case 1:	// use hamilton adams demosaicing
						mat2colorImg<float>(sequence[f], tmp_in);

						HADemosaicing(tmp_out->c1, tmp_in->c1, tmp_in->width, tmp_in->height, red_loc[0], red_loc[1]); // Hamilton-Adams implemented by Pascal Getreuer

						sequence[f] = Mat::zeros(sequence[f].rows, sequence[f].cols, CV_32FC3);
						colorImg2colorMat<Vec3f>(tmp_out, sequence[f]);
						break;

					case 2: // use opencv demosaicing
						tmp.convertTo(tmp, CV_8UC1);
						sequence[f] = Mat::zeros(tmp.rows, tmp.cols, CV_8UC3);

						int code = CV_BayerBG2RGB;
						if (red_loc[1] == 0) // y
							if (red_loc[0] == 0) // x
								code = CV_BayerBG2RGB;
							else
								code = CV_BayerGB2RGB;
						else if (red_loc[0] == 0) // x
							code = CV_BayerGR2RGB;
						else
							code = CV_BayerRG2RGB;

						cv::cvtColor(tmp, sequence[f], code); // components from second row, second column !!!!!!!!!!!!!!!!!
						sequence[f].convertTo(sequence[f], CV_32FC(sequence[f].channels()));
						break;
				}

				color_image_delete(tmp_in);
				color_image_delete(tmp_out);
			} else {
				// covert to RGB
				cv::cvtColor(sequence[f], sequence[f], CV_BGR2RGB);
			}

			if (thread_params.parameter<bool>("grayscale", "0"))
				cvtColor(sequence[f], sequence[f], CV_RGB2GRAY);

			// use only a part of the images
			if (thread_params.extent.x > 0 || thread_params.extent.y > 0) {
				sequence[f] = sequence[f].rowRange(Range(thread_params.center.y - thread_params.extent.y / 2, thread_params.center.y + thread_params.extent.y / 2));
				sequence[f] = sequence[f].colRange(Range(thread_params.center.x - thread_params.extent.x / 2, thread_params.center.x + thread_params.extent.x / 2));
			}

			// rescale image with gaussian blur to avoid anti-aliasing
			double img_scale = thread_params.parameter<double>("scale", "1.0");
			if (img_scale != 1) {
				GaussianBlur(sequence[f], sequence[f], Size(0, 0), 1 / sqrt(2 * img_scale), 1 / sqrt(2 * img_scale), BORDER_REPLICATE);
				resize(sequence[f], sequence[f], Size(0, 0), img_scale, img_scale, INTER_LINEAR);
			}

			// print to file
			char file[1024];
			sprintf(file, (acc_folder.str() + "sequence/frame_%i.png").c_str(), thread_params.sequence_start - steps * skip + f * skip);

			Mat output_img;
			if (thread_params.parameter<bool>("16bit", "0")) {
				sequence[f].convertTo(output_img, CV_16UC(sequence[f].channels()));
			} else {
				sequence[f].convertTo(output_img, CV_8UC(sequence[f].channels()), norm);
			}

			if (thread_params.parameter<bool>("grayscale", "0"))
				cv::cvtColor(output_img, output_img, CV_GRAY2BGR);	// OpenCV uses BGR
			else
				cv::cvtColor(output_img, output_img, CV_RGB2BGR);	// OpenCV uses BGR

			if (thread_params.verbosity(WRITE_FILES)) {
				imwrite(file, output_img, compression_params);
			}

			data[f] = color_image_new(sequence[f].cols, sequence[f].rows);
			if (thread_params.parameter<bool>("grayscale", "0"))
				mat2colorImg<float>(sequence[f], data[f]);
			else
				colorMat2colorImg<Vec3f>(sequence[f], data[f]);
		}

		// normalize data terms
		normalize(data, Jets + 1, thread_params);

		// compute derivatives
		for (uint32_t f = 0; f < (uint32_t) (Jets + 1); f++) {
			data_dx[f] = color_image_new(sequence[f].cols, sequence[f].rows);
			data_dy[f] = color_image_new(sequence[f].cols, sequence[f].rows);

			float deriv_filter[3] = { 0.0f, -8.0f / 12.0f, 1.0f / 12.0f };
			convolution_t *deriv = convolution_new(2, deriv_filter, 0);
			color_image_convolve_hv(data_dx[f], data[f], deriv, NULL);
			color_image_convolve_hv(data_dy[f], data[f], NULL, deriv);
			convolution_delete(deriv);
		}

		// store reference image
		sequence[0].convertTo(reference, CV_8UC(sequence[0].channels()), norm);
		double img_scale = 1.0f / (skip_pixel + 1);
		if (img_scale != 1) {
			GaussianBlur(reference, reference, Size(0, 0), 1 / sqrt(2 * img_scale), 1 / sqrt(2 * img_scale), BORDER_REPLICATE);
			resize(reference, reference, Size(0, 0), img_scale, img_scale, INTER_LINEAR);
		}

		// reference image and edges for epic flow
		float_image forward_edges;
		color_image_t *imlab = NULL;
		image_t *epic_wx = NULL, *epic_wy = NULL;
		if(thread_params.parameter<bool>("acc_epic_interpolation", "1")) {
			char image_f[1000], edges_f[1000], edges_cmd[1000];
			sprintf(image_f, (acc_folder.str() + "tmp/frame_epic_%i.png").c_str(), thread_params.sequence_start);
			sprintf(edges_f, (acc_folder.str() + "tmp/edges_%i.dat").c_str(),  thread_params.sequence_start);

			Mat tmp = reference.clone();
			// get lab image
			color_image_t* tmp_img = color_image_new(tmp.cols, tmp.rows);
			if(tmp.channels() == 1) {
				mat2colorImg<uchar>(tmp, tmp_img);
			} else
				colorMat2colorImg<Vec3b>(tmp, tmp_img);

			imlab = rgb_to_lab(tmp_img);
			color_image_delete(tmp_img);

			// get edges
			cv::cvtColor(tmp, tmp, CV_RGB2BGR);	// OpenCV uses BGR
			imwrite(image_f, tmp, compression_params);

			sprintf(edges_cmd, "matlab -nodesktop -nojvm -r \"addpath(\'%s/matlab/\'); detect_edges(\'%s\',\'%s\'); exit\"", SOURCE_PATH.c_str(), image_f, edges_f);

			system(edges_cmd);

			forward_edges = read_edges(edges_f, tmp.cols, tmp.rows);
		}


		// ######### compute smoothness weighting
		float avg_1 = thread_params.parameter<double>("img_norm_avg_1", "0"), avg_2 = thread_params.parameter<double>("img_norm_avg_2", "0"), avg_3 = thread_params.parameter<double>("img_norm_avg_3", "0"), std_1 = thread_params.parameter<double>(
				"img_norm_std_1", "1"), std_2 = thread_params.parameter<double>("img_norm_std_2", "1"), std_3 = thread_params.parameter<double>("img_norm_std_3", "1");

		// use preset local smooth weights or compute
		convolution_t *deriv;
		float deriv_filter[3] = { 0.0f, -8.0f / 12.0f, 1.0f / 12.0f };
		deriv = convolution_new(2, deriv_filter, 0);
		image_t *smooth_weight = image_new(data[0]->width, data[0]->height);
		computeSmoothnessWeight(data[0], smooth_weight, 5.0, deriv, avg_1, avg_2, avg_3, std_1, std_2, std_3, thread_params.parameter<bool>("16bit", "0"));
		convolution_delete(deriv);

		/*
		 * ################### read in ground truth ###################
		 */
		if (use_oracle && thread_params.file_gt_list.size() > 0) {
			gt = new Mat[gtFrames];
			gt_sfr = new image_t**[1];

			for (uint32_t f = 0; f < gtFrames; f++) {
				char gtF[1024];

				if (!sintel)
					sprintf(gtF, thread_params.file_gt.c_str(), thread_params.sequence_start + f);
				else {
					int sintel_frame = thread_params.sequence_start / 1000;
					int hfr_frame = f + (thread_params.sequence_start % 1000);

					while (hfr_frame < 0) {
						sintel_frame--;
						hfr_frame = 42 + hfr_frame;
					}
					while (hfr_frame > 41) {
						sintel_frame++;
						hfr_frame = hfr_frame - 42;
					}

					sprintf(gtF, thread_params.file_gt.c_str(), sintel_frame, hfr_frame);
				}

				if (access(gtF, F_OK) == -1) {
					continue;
					cerr << "Error reading " << gtF << "!" << endl;
				}

				gt[f] = readGTMiddlebury(gtF);

				// rescale gt flow
				float rescale = (1.0f * sequence[0].cols) / gt[f].cols;
				resize(gt[f], gt[f], Size(0, 0), rescale, rescale, INTER_LINEAR);
				gt[f] = rescale * gt[f];
			}
		}

		/*
		 * ################### read in ground truth occlusions ###################
		 */
		if (use_oracle && thread_params.occlusions_list.size() > 0) {
			gt_occlusions = new Mat[gtFrames];
			for (u_int32_t f = 0; f < (uint32_t) (gtFrames); f++) {
				char oocF[1024];

				if (!sintel)
					sprintf(oocF, thread_params.occlusions_list[0].c_str(), thread_params.sequence_start + f);
				else {
					int sintel_frame = thread_params.sequence_start / 1000;
					int hfr_frame = f + (thread_params.sequence_start % 1000);

					while (hfr_frame < 0) {
						sintel_frame--;
						hfr_frame = 42 + hfr_frame;
					}
					while (hfr_frame > 41) {
						sintel_frame++;
						hfr_frame = hfr_frame - 42;
					}

					sprintf(oocF, thread_params.occlusions_list[0].c_str(), sintel_frame, hfr_frame);
				}

				// check if file exists
				if (access(oocF, F_OK) != -1) {
					gt_occlusions[f] = imread(string(oocF));

					float rescale = (1.0f * sequence[0].cols) / gt_occlusions[f].cols;
					resize(gt_occlusions[f], gt_occlusions[f], Size(0, 0), rescale, rescale, INTER_CUBIC);

					// use only a part of the images
					if (thread_params.extent.x > 0 || thread_params.extent.y > 0) {
						gt_occlusions[f] = gt_occlusions[f].rowRange(Range(thread_params.center.y - thread_params.extent.y / 2, thread_params.center.y + thread_params.extent.y / 2));
						gt_occlusions[f] = gt_occlusions[f].colRange(Range(thread_params.center.x - thread_params.extent.x / 2, thread_params.center.x + thread_params.extent.x / 2));
					}

					memset(oocF, 0, 1024);
					sprintf(oocF, "%s/gt_occlusions/occ_%05i.png", acc_folder.str().c_str(), thread_params.sequence_start + f);


					// write occlusion to file
					imwrite(oocF, gt_occlusions[f]);

					if (gt_occlusions[f].channels() > 1)
						cvtColor(gt_occlusions[f], gt_occlusions[f], CV_BGR2GRAY);
					gt_occlusions[f].convertTo(gt_occlusions[f], CV_8UC1);
				} else {
					cerr << "Error reading " << oocF << "!" << endl;
				}
			}
		}

		if (!use_occlusion) {
			delete[] occlusions;
			occlusions = NULL;
		}

		/*
		 *  ########################################### data driven proposal generation ###########################################
		 */
		int owidth(sequence[0].cols), oheight(sequence[0].rows);

		int xy_incr = skip_pixel + 1;
		int xy_start = 0.5f * skip_pixel;
		int height = floor((1.0f * oheight) / xy_incr);
		int width = floor((1.0f * owidth) / xy_incr);

		int size = width * height;
		vector<vector<hypothesis*> > fb_hypotheses(size);


		// generate hypotheses from each jet estimation
		Mat consistent =  Mat::zeros(height, width, CV_32SC1);
		for (uint32_t r = 0; r < rates; r++) {
			cout << "Processing " << thread_params.jet_estimation[r] << endl;

			int r_steps = thread_params.jet_S[r] - 1;

			float ratio = (1.0f * params.jet_fps[r]) / params.jet_fps[min_fps_idx];
			uint32_t r_Jets = (ratio * Jets);
			int r_skip = (1.0f * max_fps) / thread_params.jet_fps[r];													// number of frames to skip for jet fps

			Mat *r_forward_flow = new Mat[r_Jets];
			Mat *r_backward_flow = new Mat[r_Jets];

			/*
			 * ################### read in flow fields ###################
			 */
			for (uint32_t f = 0; f < r_Jets; f++) {
				char fFlowF[1024];
				char bFlowF[1024];

				sprintf(fFlowF, ("%s" + flow_format + ".flo").c_str(), thread_params.jet_estimation[r].c_str(), (thread_params.sequence_start + f * r_steps * r_skip));
				sprintf(bFlowF, ("%s" + flow_format + "_back.flo").c_str(), thread_params.jet_estimation[r].c_str(), (thread_params.sequence_start + f * r_steps * r_skip + r_steps * r_skip));

				if (!boost::filesystem::exists(fFlowF)) {
					cerr << fFlowF << " does not exist!" << endl;
					break;
				}
				if (!boost::filesystem::exists(bFlowF)) {
					cerr << bFlowF << " does not exist!" << endl;
					break;
				}

				r_forward_flow[f] = readGTMiddlebury(fFlowF);
				r_backward_flow[f] = readGTMiddlebury(bFlowF);

				// crop images
				if (thread_params.center.x > 0) {
					// NOTE: ROWRANGE IS INDUCING ERRORS IN ACCUMULATION!!!!!!
					r_forward_flow[f] = crop(r_forward_flow[f], thread_params.center, thread_params.extent);
					r_backward_flow[f] = crop(r_backward_flow[f], thread_params.center, thread_params.extent);
				}

				// rescale image
				float rescale = (1.0f * sequence[0].cols) / r_forward_flow[f].cols;
				resize(r_forward_flow[f], r_forward_flow[f], Size(0, 0), rescale, rescale, INTER_LINEAR);
				resize(r_backward_flow[f], r_backward_flow[f], Size(0, 0), rescale, rescale, INTER_LINEAR);
				r_forward_flow[f] *= rescale;
				r_backward_flow[f] *= rescale;

				if((int) r == min_fps_idx) {
					forward_flow[f] = r_forward_flow[f];
					backward_flow[f] = r_backward_flow[f];
				}
			}

			/*
			 * ################### read in occlusions ###################
			 */
			Mat* r_occlusions = NULL;
			if(use_jet_occlusions) {
				r_occlusions = new Mat[r_Jets];
				for (uint32_t f = 0; f < (r_Jets); f++) {
					char seqF[1024];
					sprintf(seqF, "%s/occlusion/frame_%i.pbm", thread_params.jet_estimation[r].c_str(), (thread_params.sequence_start + f * r_steps * r_skip));

					if (!boost::filesystem::exists(seqF)) {
						cerr << seqF << " does not exist!" << endl;
						break;
					}

					r_occlusions[f] = imread(string(seqF), CV_LOAD_IMAGE_UNCHANGED);         // load images

					// crop images
					if (thread_params.center.x > 0)
						r_occlusions[f] = crop(r_occlusions[f], thread_params.center, thread_params.extent);

					// rescale image
					float rescale = (1.0f * sequence[0].cols) / r_occlusions[f].cols;
					resize(r_occlusions[f], r_occlusions[f], Size(0, 0), rescale, rescale, INTER_CUBIC);

					medianBlur ( r_occlusions[f], r_occlusions[f], 3 );

					if (thread_params.verbosity(WRITE_FILES)) {
						char file[1024];
						sprintf(file, (acc_folder.str() + "occlusion_%i.png").c_str(), thread_params.sequence_start + steps * skip + f * skip);
						imwrite(file, r_occlusions[f], compression_params);
					}

					// create mask out of the occlusion
					r_occlusions[f].convertTo(r_occlusions[f], CV_8UC1);
					r_occlusions[f] = (255 - r_occlusions[f]);

					if (thread_params.verbosity(VER_IN_GT)) {
						namedWindow("Occlusion");               // Create a window for display.
						imshow("Occlusion", r_occlusions[f]);
						waitKey(0);
					}

					if(min_fps_idx && (int) r == min_fps_idx) {
						occlusions[f] = r_occlusions[f];
					}
				}
			}

			/*
			 * ################### accumulate consitent trajectories ###################
			 */
			cout << "Generating hypotheses from consistent accumulations..." << endl;
			int created_hypotheses = 0;
			int rejected_hypotheses = 0;
			Mat r_consistent =  Mat::zeros(height, width, CV_32SC1);
			int r_consistent_num = 0;
			Mat* acc_cons_flow = new Mat[r_Jets];
			Mat tracked;

			double threshold = thread_params.parameter<double>("acc_consistency_threshold");
			if(use_jet_occlusions)
				tracked = accumulateConsistentBatches(acc_cons_flow, r_forward_flow, r_backward_flow, r_occlusions, r_Jets, threshold, skip_pixel, discard_inconsistent, 0);
			else
				tracked = accumulateConsistentBatches(acc_cons_flow, r_forward_flow, r_backward_flow, NULL, r_Jets, threshold, skip_pixel, discard_inconsistent, 0);

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					// only consider consistent trajectories
					if(tracked.at<int>(y,x) == (int) r_Jets) {
						consistent.at<int>(y,x) = 1;
						r_consistent.at<int>(y,x) = 1;
						r_consistent_num++;

						// get original pixel location
						int oy = y * xy_incr + xy_start;
						int ox = x * xy_incr + xy_start;

						// generate hypothesis
						double* f_x = new double[r_Jets];
						double* f_y = new double[r_Jets];
						for (int f = 0; f < tracked.at<int>(y,x); f++) {
							f_y[f] = acc_cons_flow[f].at<Vec2d>(y, x)[0];
							f_x[f] = acc_cons_flow[f].at<Vec2d>(y, x)[1];
						}

						hypothesis* hyp = new hypothesis(r_Jets, 0, tracked.at<int>(y,x), f_x, f_y, ox, oy);

						fb_hypotheses[y * width + x].push_back(new hypothesis(hyp));
						fb_hypotheses[y * width + x].back()->setLocation(ox, oy);
						fb_hypotheses[y * width + x].back()->jet_est = r;

						fb_hypotheses[y * width + x].back()->adaptFPS(Jets);

						// compute energy
						fb_hypotheses[y * width + x].back()->setOcclusions(forward_flow, backward_flow, occlusion_threshold, occlusion_fb_threshold);
						fb_hypotheses[y * width + x].back()->energy = addJC(fb_hypotheses[y * width + x].back(), forward_flow, acc_jc, acc_cv, phi_d, thread_params, occlusions)
								+ addBCGC(fb_hypotheses[y * width + x].back(), data, data_dx, data_dy, acc_bc, acc_gc, skip_pixel, thread_params, occlusions)
								+ addOC(fb_hypotheses[y * width + x].back(), acc_occ, acc_temporal_occ, thread_params)
								+ weight_jet_estimation[r];

						created_hypotheses++;


						delete hyp;
					} else
						rejected_hypotheses++;
				}
			}

			// obtain consistent maps
			removeSmallSegments(r_consistent, 0.1, 100);

			/*
			 * ################### epic flow interpolation in occluded regions ###################
			 */
			if(thread_params.parameter<bool>("acc_epic_interpolation", "1")) {
				vector<hypothesis*> epic_hypotheses(height * width, NULL);

				// epic interpolation using consistent flow
				for (uint32_t j = 0; j < r_Jets; j++) {
					float_image matches = empty_image(float, 4, r_consistent_num);
					int m = 0;
					for (int y = floor(0.5f * epic_skip); y < height; y+=epic_skip) {
						for (int x = floor(0.5f * epic_skip); x < width; x+=epic_skip) {
							if(r_consistent.at<int>(y,x) == 1) {
								matches.pixels[4*m] = x;
								matches.pixels[4*m + 1] = y;
								matches.pixels[4*m + 2] = x + acc_cons_flow[j].at<Vec2d>(y,x)[1] / (skip_pixel + 1);
								matches.pixels[4*m + 3] = y + acc_cons_flow[j].at<Vec2d>(y,x)[0] / (skip_pixel + 1);
								m++;
							}

						}
					}
					matches.ty = m;
					cout << "Using " << m << " Matches!" << endl;

					// prepare variables
					image_t *wx = image_new(imlab->width, imlab->height), *wy = image_new(imlab->width, imlab->height);
					epic(wx, wy, imlab, &matches, &forward_edges, &epic_params, 1);

					if(thread_params.parameter<bool>("acc_epic_interpolation", "1")) {
						for (int y = 0; y < height; y++) {
							for (int x = 0; x < width; x++) {
								int oy = y * xy_incr + xy_start;
								int ox = x * xy_incr + xy_start;

								// init epic hypotheses
								if(epic_hypotheses[y * width + x] == NULL) {
									epic_hypotheses[y * width + x] = new hypothesis(r_Jets, 0, r_Jets, new double[r_Jets], new double[r_Jets], ox, oy);
									epic_hypotheses[y * width + x]->jet_est = r;
								}

								// set flow
								epic_hypotheses[y * width + x]->flow_x[j] = wx->data[y * wx->stride + x] * (skip_pixel + 1);
								epic_hypotheses[y * width + x]->flow_y[j] = wy->data[y * wy->stride + x] * (skip_pixel + 1);

								// finish hypothesis
								if(j == r_Jets - 1) {
									epic_hypotheses[y * width + x]->adaptFPS(Jets);

									epic_hypotheses[y * width + x]->setOcclusions(forward_flow, backward_flow, occlusion_threshold, occlusion_fb_threshold);
									epic_hypotheses[y * width + x]->energy = addJC(epic_hypotheses[y * width + x], forward_flow, acc_jc, acc_cv, phi_d, thread_params, occlusions)
																			+ addBCGC(epic_hypotheses[y * width + x], data, data_dx, data_dy, acc_bc, acc_gc, skip_pixel, thread_params, occlusions)
																			+ addOC(epic_hypotheses[y * width + x], acc_occ, acc_temporal_occ, thread_params)
																			+ weight_jet_estimation[r];

									fb_hypotheses[y * width + x].push_back(epic_hypotheses[y * width + x]);
								}
							}
						}
					}

					// write flow image to file
					if(thread_params.verbosity(WRITE_FILES)) {
						Mat floImg = flowColorImg(wx, wy, params.verbosity(VER_CMD));
						if(floImg.data && !acc_folder.str().empty()) {
							stringstream flowF;
							flowF << acc_folder.str() <<  "tmp/epic_" << params.jet_fps[r] << "fps_" << thread_params.sequence_start << "_" << j << ".png";

							imwrite((flowF.str()), floImg);
						}
					}

					free(matches.pixels);
					if((int) r == min_fps_idx && j == r_Jets - 1) {
						epic_wx = wx;
						epic_wy = wy;
						image_mul_scalar(epic_wx, (skip_pixel + 1));
						image_mul_scalar(epic_wy, (skip_pixel + 1));
					} else {
						image_delete(wx);
						image_delete(wy);
					}
				}
			}

			if (thread_params.verbosity(VER_CMD))
				cout << created_hypotheses << " trajectory hypotheses generated! (" << rejected_hypotheses << " rejected)" << endl;

			// cleanup
			delete[] acc_cons_flow;
			delete[] r_forward_flow;
			delete[] r_backward_flow;
			if(r_occlusions != NULL) delete[] r_occlusions;
		}

		/*
		 *  ###################### dense tracking formulation ######################
		 */

		time_t unary_start, unary_end;
		time_t opt_start, opt_end;

		// reset data term time
		dt_warp_time = 0;
		dt_med_time = 0;
		dt_sum_time = 0;

		// initialize MRF object
		uint32_t numVariables = height * width;				// maximum number of variables
		uint32_t factors = 0;
		Mat selected_hyp = Mat::zeros(height, width, CV_32SC1);
		/*
		 *  ###################### discret optimization: flow reasoning ######################
		 */
		for (int p_it = 0; p_it < alternate; p_it++) {
			default_random_engine generator(seed + p_it);

			if (p_it > 0) {
				// keep only top x
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						int idx = y * width + x;

						vector<hypothesis*> temp = fb_hypotheses[idx];
						fb_hypotheses[idx].clear();

						// add best hypotheses from last iteration
						int last = selected_hyp.at<int>(y, x);
						if (last >= 0 && (uint32_t) last < temp.size()) {
							fb_hypotheses[idx].push_back(temp[last]);
							temp[last] = NULL;
						}

						// sort in ascending order
						sort(temp.begin(), temp.end(), compareHypotheses);

						// add to get top x
						uint32_t h = 0;
						for (; h < temp.size(); h++) {
							if(temp[h] == NULL)
								continue;

							if(fb_hypotheses[idx].size() <= perturb_keep)
								fb_hypotheses[idx].push_back(temp[h]);
							else
								// clean up the rest
								delete temp[h];
						}
					}
				}
			} else {
				// SORT ACCORDING DATATERM!
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						int idx = y * width + x;

						// sort all hypothesis besides outlier hypothesis
						if (fb_hypotheses[idx].size() > 1) {
							sort(fb_hypotheses[idx].begin(), fb_hypotheses[idx].end(), compareHypotheses);
						}
					}
				}
			}

			/*
			 *  ########################################### hypotheses from neighbors ###########################################
			 */
			{
				cout << "neighbors" << endl;
				vector<vector<hypothesis*> > tmp = fb_hypotheses;;

				clock_t start_nn = clock();
				// build nearest neighbor tree
				int counter2 = 0;
				int counter4 = 0;
				for (int y = 1; y < height; y++) {
					for (int x = 1; x < width; x++) {
						// only use consistent estimates for first draw
						if(consistent.at<int>(y,x) == 1 || p_it > 0) {
							if((y - 1) % nn_skip1 == 0 && (x - 1) % nn_skip1 == 0) counter2++;
							if((y - 2) % nn_skip2 == 0 && (x - 2) % nn_skip2 == 0) counter4++;
						}
					}
				}

				vector<flann::Matrix<float>> dataset(2);
				dataset[0] = flann::Matrix<float>(new float[counter2 * 2], counter2, 2);
				dataset[1] = flann::Matrix<float>(new float[counter4 * 2], counter4, 2);
				counter2 = 0;
				counter4 = 0;
				for (int y = 1; y < height; y++) {
					for (int x = 1; x < width; x++) {
						// only use consistent estimates for first draw
						if(consistent.at<int>(y,x) == 1 || p_it > 0) {
							if((y - 1) % nn_skip1 == 0 && (x - 1) % nn_skip1 == 0) {
								dataset[0].ptr()[counter2 * 2] = x;
								dataset[0].ptr()[counter2 * 2 + 1] = y;
								counter2++;
							}

							if((y - 2) % nn_skip2 == 0 && (x - 2) % nn_skip2 == 0) {
								dataset[1].ptr()[counter4 * 2] = x;
								dataset[1].ptr()[counter4 * 2 + 1] = y;
								counter4++;
							}
						}
					}
				}

				vector<flann::Index<flann::L2<float> >*> nonempty;
				nonempty.push_back(new flann::Index<flann::L2<float> >(flann::KDTreeSingleIndexParams()));
				nonempty.push_back(new flann::Index<flann::L2<float> >(flann::KDTreeSingleIndexParams()));
				nonempty[0]->buildIndex(dataset[0]);
				nonempty[1]->buildIndex(dataset[1]);

				cout << "First NN tree " << counter2 << endl << "Second NN tree " << counter4 << endl;

				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {

						int i = y * width + x;

						int oy = y * xy_incr + xy_start;
						int ox = x * xy_incr + xy_start;

						for(uint32_t t = 0; t < nonempty.size(); t++) {
							// compare to existent trajectories (nearest neighbors)
							flann::Matrix<float> query(new float[2], 1, 2);
							std::vector<std::vector<int> > indices;
							std::vector<std::vector<float> > dists;
							std::vector<double> prob;
							std::vector<double> cum;

							query.ptr()[0] = x;
							query.ptr()[1] = y;

							int found_nn = 0;
							if(draw_nn_from_radius) {
								found_nn = nonempty[t]->radiusSearch(query, indices, dists, (t + 1) * (hyp_neigh_radius), flann_params_p);
								if(found_nn < 50) {
									found_nn = nonempty[t]->knnSearch(query, indices, dists, 50, flann_params_p);
								}
							} else {
								found_nn = nonempty[t]->knnSearch(query, indices, dists, hyp_neigh_draws, flann_params_p);
							}

							// remove x,y
							if(fb_hypotheses[i].size() > 0) {
								std::vector<std::vector<int> > tmp_indices = indices;
								std::vector<std::vector<float> > tmp_dists = dists;
								indices[0].clear();
								dists[0].clear();
								int tmp_found_nn = 0;
								for(int i = 0; i < found_nn; i++) {
									if(dataset[t].ptr()[2 * tmp_indices[0][i]] != x && dataset[t].ptr()[2 * tmp_indices[0][i] + 2] != y) {
										indices[0].push_back(tmp_indices[0][i]);
										dists[0].push_back(tmp_dists[0][i]);
										tmp_found_nn++;
									}
								}
								found_nn = tmp_found_nn;
							}

							std::uniform_real_distribution<double> uniform1(0, 1);
							uniform_int_distribution<int> uniform2(0, found_nn - 1);

							int tryouts = 0;
							// draw half of the neighbors
							while(tryouts < hyp_neigh_tryouts && (fb_hypotheses[i].size() - tmp[i].size()) < (t + 1) * hyp_neigh) {
								tryouts++;

								int ridx = uniform2(generator);

								int nx = x, ny = y, ni = ny * width + nx;
								nx = dataset[t].ptr()[indices[0][ridx] * 2];
								ny = dataset[t].ptr()[indices[0][ridx] * 2 + 1];
								ni = ny * width + nx;

								hypothesis* top_n_h = NULL;
								// select best hypothesis at nx,ny!
								// hypotheses are sort in ascending order (the last selected hypothesis is the first, the others are sorted using the data term!)
								top_n_h = new hypothesis(tmp[ni][0]);
								top_n_h->setLocation(ox, oy);
								top_n_h->setOcclusions(forward_flow, backward_flow, occlusion_threshold, occlusion_fb_threshold);
								top_n_h->energy = addJC(top_n_h, forward_flow, acc_jc, acc_cv, phi_d, thread_params, occlusions) + addBCGC(top_n_h, data, data_dx, data_dy, acc_bc, acc_gc, skip_pixel, thread_params, occlusions)
										+ addOC(top_n_h, acc_occ, acc_temporal_occ, thread_params)
										+ weight_jet_estimation[top_n_h->jet_est];

								if(top_n_h != NULL) {
									// check if a similar hypothesis does not already exists
									bool discard = false;
									for (uint32_t h = 0; h < fb_hypotheses[i].size() && !discard; h++) {
											// only store dissimilar or longer trajectories
											discard = (fb_hypotheses[i][h]->compare((*top_n_h), traj_sim_thres, traj_sim_method) >= 0);
									}

									if(!discard) {
											fb_hypotheses[i].push_back(top_n_h);
									} else
											delete top_n_h;
								}
							}

							if(fb_hypotheses[i].size() == 0)
								cout << found_nn << endl;
						}
					}
				}

				delete dataset[0].ptr();
				delete dataset[1].ptr();
				delete nonempty[0];
				delete nonempty[1];

				clock_t end_nn = clock();
				cout << "NN propagation took " << (double) (end_nn-start_nn) / CLOCKS_PER_SEC << "secs" << endl;
			}

			/*
			 *  ########################################### non maximum suppression ###########################################
			 */
			int all_h = 0;
			int remaining_h = 0;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx = y * width + x;

					all_h += fb_hypotheses[idx].size();

					if (fb_hypotheses[idx].size() > 1) {
						vector<hypothesis*> nms_hypotheses;

						// resort after adding neighboring hypotheses!
						// hypotheses are sort in ascending order (the last selected hypothesis is the first, the others are sorted using the data term!)
						if(p_it > 0)
							sort(fb_hypotheses[idx].begin() + 1, fb_hypotheses[idx].end(), compareHypotheses);
						else
							sort(fb_hypotheses[idx].begin(), fb_hypotheses[idx].end(), compareHypotheses);

						nms_hypotheses.push_back(fb_hypotheses[idx][0]);

						// iterate over all other hypotheses
						for (uint32_t nh = 1; nh < fb_hypotheses[idx].size(); nh++) {
							bool discard = false;

							// compare to already added hypotheses
							for (uint32_t h = 0; h < nms_hypotheses.size(); h++) {
								if (fb_hypotheses[idx][nh]->distance(*nms_hypotheses[h], traj_sim_method) < traj_sim_thres)
									discard = true;
							}

							// add or delete
							if (!discard) {
								nms_hypotheses.push_back(fb_hypotheses[idx][nh]);
							} else {
								delete fb_hypotheses[idx][nh];
								fb_hypotheses[idx][nh] = NULL;
								break;
							}
						}

						fb_hypotheses[idx].clear();
						fb_hypotheses[idx] = nms_hypotheses;
					}

					remaining_h += fb_hypotheses[idx].size();
				}
			}

			if (thread_params.verbosity(VER_CMD))
				cout << endl << "From " << all_h << " " << remaining_h << " hypotheses are remaining after non maxima suppression!" << endl;

			/*
			 *  ########################################### select minimal hypothesis ###########################################
			 */
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int i = y * width + x;

					if (fb_hypotheses[i].empty())
						continue;

					vector<hypothesis*> sorted_hyp = fb_hypotheses[i];
					sort(sorted_hyp.begin(), sorted_hyp.end(), compareHypotheses);
				}
			}

			// initialize MRF object
			numVariables = height * width;				// maximum number of variables
			factors = 0;

			// create MRF object
			MRFEnergy<TypeGeneral>* mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
			vector<MRFEnergy<TypeGeneral>::NodeId> nodes(height * width);
			TypeGeneral::REAL energy, lowerBound;

			cout << "Generate MRF object..." << endl;

			// data term: add unary nodes
			time(&unary_start);
			cout << "	adding unary potentials ..." << flush;

			// everything visible in first frame
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx = y * width + x;

					uint32_t labels = fb_hypotheses[idx].size();

					if(labels == 0) {
						throw std::out_of_range("One pixel without hypotheses! Please add outlier hypotheses!");
					}

					double* e_mex = new double[labels];

					// add data terms
					for (uint32_t h = 0; h < labels; h++) {
						e_mex[h] = fb_hypotheses[idx][h]->energy;
					}

					if (thread_params.verbosity(VER_CMD)) {
						if (x == 120 && y == 90) {
							cout << "Unary: P(" << x << ", " << y << ")" << endl;
							for (uint32_t h = 0; h < labels; h++)
								cout << " 	" << h;
							cout << endl;

							for (uint32_t h = 0; h < labels; h++)
								cout << " 	" << e_mex[h];
							cout << endl << endl;
						}
					}

					nodes[idx] = mrf->AddNode(TypeGeneral::LocalSize(labels), TypeGeneral::NodeData(e_mex));
					delete[] e_mex;

					factors++;
				}
			}

			time(&unary_end);
			cout << " took " << difftime(unary_end, unary_start) << endl;

			#pragma omp critical
			{
				avg_unary_time += difftime(unary_end, unary_start);
			}

			time(&unary_start);
			cout << "	adding pairwise potentials ..." << flush;

			// adding pairwise
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx1 = y * width + x;
					int oidx1 = (y * xy_incr + xy_start) * owidth + x * xy_incr + xy_start;

					uint32_t labels1 = fb_hypotheses[idx1].size();

					if (labels1 == 0)
						continue;

					vector<int> oneighbors;
					vector<int> neighbors;
					if (x + 1 < width) {
						neighbors.push_back(y * width + x + 1);
						oneighbors.push_back((y * xy_incr + xy_start) * owidth + (x + 1) * xy_incr + xy_start);
					}
					if (y + 1 < height) {
						neighbors.push_back((y + 1) * width + x);
						oneighbors.push_back(((y + 1) * xy_incr + xy_start) * owidth + x * xy_incr + xy_start);
					}

					vector<int>::iterator oidx2 = oneighbors.begin();
					for (vector<int>::iterator idx2 = neighbors.begin(); idx2 != neighbors.end(); idx2++, oidx2++) {
						uint32_t labels2 = fb_hypotheses[*idx2].size();

						if (labels2 == 0)
							continue;

						// create energy matrix
						TypeGeneral::REAL *P = new TypeGeneral::REAL[labels1 * labels2];

						for (uint32_t h1 = 0; h1 < labels1; h1++) {
							for (uint32_t h2 = 0; h2 < labels2; h2++) {
								float dist = outlier_beta;
								float smooth_occ = 0;

								hypothesis* hyp_p1 = fb_hypotheses[idx1][h1];
								hypothesis* hyp_p2 = fb_hypotheses[*idx2][h2];

								// similarity of hypotheses
								dist = hyp_p1->distance(*hyp_p2, traj_sim_method);

								// spatial smoothness for occlusion
								for (uint32_t j = 0; j < Jets + 1; j++)
									if (hyp_p1->occluded(j) != hyp_p2->occluded(j))
										smooth_occ++;

								P[h2 * labels1 + h1] = (smooth_weight->data[oidx1] + smooth_weight->data[*oidx2]) * (acc_beta * dist + acc_spatial_occ * smooth_occ);

							}
						}

						if (thread_params.verbosity(VER_CMD)) {
							if (x == 120 && y == 90) {
								// ##### DEBUG EXTENDED OUTPUT
								if (thread_params.verbosity(VER_CMD)) {
									cout << "Pairwise: SP(" << x << ", " << y << "), SP(" << *idx2 << ")" << endl;
									cout << " 	";
									for (uint32_t h_2 = 0; h_2 < labels2; h_2++) {
										cout << "h" << h_2 << "	";
									}
									cout << endl;

									for (uint32_t h_1 = 0; h_1 < labels1; h_1++) {
										cout << " h" << h_1 << "	";
										for (uint32_t h_2 = 0; h_2 < labels2; h_2++) {
											cout << P[h_2 * labels1 + h_1] << "	";
										}
										cout << endl;
									}
									cout << endl;
								}
							}
						}

						mrf->AddEdge(nodes[idx1], nodes[*idx2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, P));
						delete[] P;

						factors++;
					}
				}
			}
			time(&unary_end);
			cout << " took " << difftime(unary_end, unary_start) << endl;

			#pragma omp critical
			{
				avg_pairw_time += difftime(unary_end, unary_start);
			}

			cout << "Variables:\t" << numVariables << endl;
			cout << "Factors:\t" << factors << endl;

			// run discrete optimization TRW-S
			cout << "Run discrete optimization..." << endl;
			time(&opt_start);

			mrf->SetAutomaticOrdering();

			if (options.m_method == 0) {
				mrf->Minimize_TRW_S(options, lowerBound, energy);

				time(&opt_end);

				printf("TRW-S finished. Time: %i\n", (int) difftime(opt_end, opt_start));
			} else {
				mrf->Minimize_BP(options, energy);
				lowerBound = std::numeric_limits<double>::signaling_NaN();

				time(&opt_end);

				printf("BP finished. Time: %i\n", (int) difftime(opt_end, opt_start));
			}

			#pragma omp critical
			{
				avg_optimization_time += difftime(opt_end, opt_start);
			}

			// get solutions
			Mat oracle_selection = Mat::zeros(height, width, CV_8UC3);
			Mat oracle_present = Mat::zeros(height, width, CV_8UC3);
			Mat occlusion_map = Mat::zeros(height, width, CV_32FC1);
			Mat flow = Mat::zeros(height, width, CV_64FC2);

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx = y * width + x;
					int h = -1;

					if (fb_hypotheses[idx].size() > 0)
						h = (int) (mrf->GetSolution(nodes[idx]));

					// remember best hypothesis
					selected_hyp.at<int>(y, x) = h;

					// get flow of best hypothesis
					flow.at<Vec2d>(y, x)[1] = fb_hypotheses[idx][h]->u(Jets - 1) / xy_incr;
					flow.at<Vec2d>(y, x)[0] = fb_hypotheses[idx][h]->v(Jets - 1) / xy_incr;

					// get occlusion map
					occlusion_map.at<float>(y, x) = fb_hypotheses[idx][h]->occluded(0);
					for (uint32_t f = 0; f < Jets; f++)
						occlusion_map.at<float>(y, x) = max((int) occlusion_map.at<float>(y, x), fb_hypotheses[idx][h]->occluded(f + 1));
				}
			}

			#pragma omp critical
			{
				numVariablesStream << "\t" << numVariables;
				factorsStream << "\t" << factors;
			}

			delete mrf;

			/*
			 *  ########################################### visualization and store accumulated flow ###########################################
			 */
			if (p_it == (alternate - 1)) {
				Mat flow_img = flowColorImg(flow, thread_params.verbosity(VER_CMD));

				 // Check for invalid input
				if (!flow_img.data) {
					cout << "No flow for frame " << thread_params.sequence_start << std::endl;
					continue;
				}

				// write final estimation
				char occF[1024];
				sprintf(occF, (acc_folder.str() + "/occlusions/frame_%i.pbm").c_str(), thread_params.sequence_start);

				occlusion_map.convertTo(occlusion_map, CV_8UC1, 255);
				imwrite(occF, occlusion_map);

				char forward_flow_file[1024];
				if (!sintel)
					sprintf(forward_flow_file, (acc_folder.str() + "/" + flow_format).c_str(), thread_params.sequence_start);
				else
					sprintf(forward_flow_file, (acc_folder.str() + "/" + flow_format).c_str(), thread_params.sequence_start, 0);

				imwrite((string(forward_flow_file) + "_vis.png"), flow_img);

				writeFlowMiddlebury(flow, (string(forward_flow_file) + ".flo"));
			}
		}

		cout << endl << "Writing results to: " << acc_folder.str() << endl << endl;

		//----------------------------------
		// ########################################### final clean up ###########################################
		//----------------------------------
		for (uint32_t i = 0; i < fb_hypotheses.size(); i++) {
			for (uint32_t h = 0; h < fb_hypotheses[i].size(); h++) {
				delete fb_hypotheses[i][h];
			}
		}
		fb_hypotheses.clear();
		fb_hypotheses = vector<vector<hypothesis*> >();

		for (uint32_t f = 0; f < (Jets + 1); f++) {
			color_image_delete(data[f]);
			color_image_delete(data_dx[f]);
			color_image_delete(data_dy[f]);
		}
		delete[] data;
		delete[] data_dx;
		delete[] data_dy;
		image_delete(smooth_weight);

		if(imlab != NULL) {
			color_image_delete(imlab);
			free(forward_edges.pixels);
		}
		if(epic_wx != NULL) image_delete(epic_wx);
		if(epic_wy != NULL) image_delete(epic_wy);

		delete[] sequence;
		if (gt_sfr != NULL) {
			image_delete(gt_sfr[0][0]);
			image_delete(gt_sfr[0][1]);
			delete[] gt_sfr[0];
			delete[] gt_sfr;
		}
		if (gt != NULL) 			delete[] gt;
		if (gt_occlusions != NULL) 	delete[] gt_occlusions;
		if (occlusions != NULL) 	delete[] occlusions;
		delete[] forward_flow;
		delete[] backward_flow;

		/*
		 *  write results and infos to file
		 */
		if (!params.output.empty()) {
			ofstream infos;
			infos.open((acc_folder.str() + "/result.info").c_str());
			infos << "# Discrete optimization file\n\n";
			infos << "Warping took " << dt_warp_time << "s.\n";
			infos << "Median took " << dt_med_time << "s.\n";
			infos << "Data term computation took " << dt_sum_time << "s.\n";
			infos << "Adding unary potentials took " << avg_unary_time << "s.\n";
			infos << "Adding pairwise potentials took " << avg_pairw_time << "s.\n";
			infos << "Run discrete optimization took " << avg_optimization_time << "s.\n\n";
			infos << "Discrete Optimization:\n";
			infos << "\tVariables:\t" << numVariablesStream.str() << "\n";
			infos << "\tFactors:\t" << factorsStream.str() << "\n\n";
			infos.close();
		}
	}

	if (phi_d != NULL)
		delete phi_d;
	if (phi_d != NULL)
		delete phi_s;


	cout << "Done!" << endl;

	return 0;
}

