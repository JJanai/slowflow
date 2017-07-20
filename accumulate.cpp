/*
 * accumulate.cpp
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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include "epic.h"
#include "image.h"
#include "io.h"
#include "variational_mt.h"
#include "utils/edge_detector.h"
#include "utils/utils.h"

#include "../multi_frame_flow/multi_frame_optical_flow.h"
#include "../multi_frame_flow/utils/parameter_list.h"

#include "../libs/middlebury_devkit/cpp/colorcode.h"
#include "../libs/middlebury_devkit/cpp/flowIO.h"

using namespace std;
using namespace cv;

void setDefaultVariational(ParameterList& params) {
    // general
    params.insert("verbose", "0", true);
    params.insert("threads", "1", true);

//    params.insert("format", "img_%04d.tif", true);

    params.insert("stats", "0", true);
    params.insert("S", "2", true);

    params.insert("scale", "1.0f", true);

    params.insert("dataterm", "1");
    params.insert("smoothing", "1");

    params.insert("epic_alpha", "1.0f");
    params.insert("epic_gamma", "0.72f", true);
    params.insert("epic_delta", "0.0f", true);

    params.insert("layers", "1", true);
    params.insert("p_scale", "0.9f", true);;

    params.insert("niter_alter", "1", true);
    params.insert("niter_outer", "5", true);
    params.insert("thres_outer", "1e-5", true);
    params.insert("niter_inner", "1", true);
    params.insert("thres_inner", "1e-5", true);
    params.insert("niter_solver", "30", true);
    params.insert("sor_omega", "1.9f", true);

    params.insert("robust_color", "1", true);
    params.insert("robust_color_eps", "0.001", true);
    params.insert("robust_color_truncation", "0.5", true);
    params.insert("robust_reg", "1", true);
    params.insert("robust_reg_eps", "0.001", true);
    params.insert("robust_reg_truncation", "0.5", true);

    params.insert("gradient_sigmoid", "5.0f", true);
}

int main(int argc, char **argv) {
	/*
	 *  getting config file location
	 */
#define isarg(key)  !strcmp(a,key)
	string file;
	string input_path = "";
	string output_path = "";
	if (argc > 1) {
		file = string(argv[1]);

		if(boost::filesystem::exists(file)) {
			printf("using parameters %s\n", file.c_str());
		} else {
			cerr << "usage: ./accumulate [cfg-file] -input_path [path] -output_path [path] {optional:Frames}" << endl;
			return -1;
		}

		int selected_jet = -1;
		int current_arg = 1;
		while(current_arg < argc ){
			const char* a = argv[current_arg++];
			if(a[0] != '-') {
				continue;
			}

			if( isarg("-h") || isarg("-help") )
				cerr << "usage: ./accumulate [cfg-file] -input_path [path] -output_path [path] {optional:Frames}" << endl;
			else if( isarg("-input_path") )
				input_path = string(argv[current_arg++]);
			else if( isarg("-output_path") )
				output_path = string(argv[current_arg++]);
			else{
				fprintf(stderr, "unknown argument %s", a);
				cerr << "usage: ./accumulate [cfg-file] -input_path [path] -output_path [path] {optional:Frames}" << endl;
				exit(1);
			}
		}
	} else {
		cerr << "usage: ./accumulate [cfg-file] -input_path [path] -output_path [path] {optional:Frames}" << endl;
		return -1;
	}

	/*
	 *  read in parameters and print config
	 */
	ParameterList params;

	params.read(file);
	string path = file.substr(0, file.find_last_of('/'));

	bool sintel = params.parameter<bool>("sintel", "0");

	// add input path and output path
	params.setParameter("flow_file", input_path + params.parameter("flow_file"));
	for(int i = 0; i < params.file_list.size(); i++)
		params.file_list[i] = input_path + params.file_list[i];
	for(int i = 0; i < params.file_gt_list.size(); i++)
		params.file_gt_list[i] = input_path + params.file_gt_list[i];
	params.output = path + "/";

	/*
	 * decompose sequence in subsets and adjust number of frames if necessary
	 */
	uint32_t Jets = params.Jets;

	if (!params.exists("S") || params.parameter<int>("S") < 2)
		params.setParameter<int>("S", 2);

	int steps = params.parameter<int>("S") - 1;

	if(Jets == 0) {
		Jets = floor(params.F / steps);
		params.Jets = Jets;
	}


//	int batch_size = floor((1.0f * params.parameter<int>("max_fps")) / (params.parameter<int>("ref_fps")));
	int batch_size = floor((1.0f * params.parameter<int>("max_fps")) / (params.parameter<int>("jet_fps")));
//	int batches = floor((1.0f * params.Jets) * batch_size);
	Jets = floor(batch_size / steps);
	int batches = params.parameter<int>("batches", "1");

	int skip = floor((1.0f * params.parameter<int>("max_fps")) / (params.parameter<int>("jet_fps")));
	if(params.exists("frames")) {
		// decompose sequence in batches
		skip = ceil(1.0f * batch_size / (params.parameter<int>("frames") - 1.0));			// with ceil we will estimate too far by fractions
		Jets = floor(1.0f * Jets / skip);
	}

	// iterate over experiments
	for(uint32_t exp = 0; exp < params.experiments(); exp++) {
		ParameterList exp_params;

		exp_params = ParameterList(params);
		params.nextExp();

		// get scaling factor
		double scale = params.parameter<float>("scale", "1.0");

		string flow_format = exp_params.parameter<string>("flow_format", "frame_%i");

		// create folder for accumulated batches
		stringstream acc_folder;
		acc_folder << exp_params.output << "accumulation/";
		boost::filesystem::create_directories(acc_folder.str());
		boost::filesystem::create_directories(acc_folder.str() + "gt_occlusions/");          // accumulate folder

		exp_params.insert("accumulation", acc_folder.str() , true);
		exp_params.Jets = batches;
		stringstream news;
		news << batch_size*steps + 1;
		exp_params.setParameter("S", news.str());

		exp_params.print();

		ofstream infos;
		infos.open((acc_folder.str() + "config.cfg").c_str());
		infos << "# SlowFlow variational estimation\n";
		infos << exp_params;
		infos.close();

		uint32_t num_sources = params.file_list.size();

		for (uint32_t source = 0; source < num_sources; source++) {
			for(int b = 0; b < batches; b++) {
				if(b > 0)
					exp_params.sequence_start_list[source] += Jets * skip * steps;

				// read in sequence, forward and backward flow, and segmentation
				Mat* gt = NULL;  	 	 // plus reference frame
				Mat* gt_occlusions = NULL;

				Mat *forward_flow = new Mat[Jets];
				Mat *backward_flow = new Mat[Jets];

				/*
				 * ################### read in ground truth ###################
				 */
				if(exp_params.file_gt.size() > 0) {
					gt = new Mat[batch_size];
					for (uint32_t f = 0; f < (batch_size); f++) {
						char gtF[200];

						if(!sintel)
							sprintf(gtF, exp_params.file_gt_list[source].c_str(), exp_params.sequence_start_list[source] + f);
						else {
							int sintel_frame = exp_params.sequence_start_list[source] / 1000;
							int hfr_frame = f + (exp_params.sequence_start_list[source] % 1000);

							while(hfr_frame < 0) {
								sintel_frame--;
								hfr_frame = 42 + hfr_frame;
							}
							while(hfr_frame > 41) {
								sintel_frame++;
								hfr_frame = hfr_frame - 42;
							}

							sprintf(gtF, exp_params.file_gt_list[source].c_str(), sintel_frame, hfr_frame);
						}

						if (access(gtF, F_OK) == -1) {
							continue;
							cerr << "Error reading "<< gtF << "!" << endl;
						}
						cout << "Reading "<< gtF << "!" << endl;

						gt[f] = readGTMiddlebury(gtF);

						// rescale gt flow
						resize(gt[f], gt[f], Size(0, 0), scale, scale, INTER_LINEAR);
						gt[f] = scale * gt[f];
					}
				}

				/*
				 * ################### read in ground truth occlusions ###################
				 */
				if(exp_params.occlusions_list.size() > 0) {
					gt_occlusions = new Mat[batch_size];
					for (u_int32_t f = 0; f < (uint32_t) (batch_size); f++) {
						char oocF[500];

						if(!sintel)
							sprintf(oocF, exp_params.occlusions_list[source].c_str(), exp_params.sequence_start_list[source] + f);
						else {
							int sintel_frame = exp_params.sequence_start_list[source] / 1000;
							int hfr_frame = f + (exp_params.sequence_start_list[source] % 1000);

							while(hfr_frame < 0) {
								sintel_frame--;
								hfr_frame = 42 + hfr_frame;
							}
							while(hfr_frame > 41) {
								sintel_frame++;
								hfr_frame = hfr_frame - 42;
							}

							sprintf(oocF, exp_params.occlusions_list[source].c_str(), sintel_frame, hfr_frame);
						}

						// read file only if already processed by first experiment
						if(access( oocF, F_OK ) != -1) {
							cout << "Reading "<< oocF << "!" << endl;

							gt_occlusions[f] = imread(string(oocF));

							resize(gt_occlusions[f], gt_occlusions[f], Size(0, 0), scale, scale, INTER_CUBIC);

							memset(oocF, 0, 500);
							sprintf(oocF, "%s/gt_occlusions/occ_%05i.png", acc_folder.str().c_str(), exp_params.sequence_start + f);
							// read file only if already processed by first experiment
							if(access( oocF, F_OK ) == -1) {
								// write flow and flow image to file
								imwrite(oocF, gt_occlusions[f]);
							}

							if(gt_occlusions[f].channels() > 1)
								cvtColor(gt_occlusions[f], gt_occlusions[f], CV_BGR2GRAY);
							gt_occlusions[f].convertTo(gt_occlusions[f], CV_8UC1);
						} else {
							cerr << "Error reading "<< oocF << "!" << endl;
						}
					}
				}


				char source_str[1024];
				if(params.name_list.size() > 1)
					sprintf(source_str, "%s%02i_", params.name_list[source].c_str(), params.id(source));
				else
					source_str[0] = '\0';

				/*
				 * ################### read in flow fields ###################
				 */
				for (uint32_t f = 0; f < Jets; f++) {
					char fFlowF[200];
					char bFlowF[200];

	//				if (exp_params.exists("flow_file")) {
	//					if(!sintel) {
	//						sprintf(fFlowF, (exp_params.parameter("flow_file") + ".flo").c_str(), exp_params.sequence_start_list[source] + f * steps);
	//						sprintf(bFlowF, (exp_params.parameter("flow_file") + "_back.flo").c_str(), exp_params.sequence_start_list[source] + f * steps + steps);
	//					} else {
	//						sprintf(fFlowF, (exp_params.parameter("flow_file") + ".flo").c_str(), exp_params.sequence_start_list[source] + f * steps, 0);
	//						sprintf(bFlowF, (exp_params.parameter("flow_file") + "_back.flo").c_str(), exp_params.sequence_start_list[source] + f * steps + steps, 0);
	//					}
	//				} else {
	//					sprintf(fFlowF, "%sframe_%i.flo", exp_params.output.c_str(), (exp_params.sequence_start_list[source] + f * steps));
	//					sprintf(bFlowF, "%sframe_%i_back.flo", exp_params.output.c_str(), (exp_params.sequence_start_list[source] + f * steps + steps));
	//				}
					string hfr = "";
					if (boost::filesystem::exists( exp_params.output + "hfr/" ))
						hfr = "hfr/";

					sprintf(fFlowF, ("%s%s" + flow_format + ".flo").c_str(), (exp_params.output + hfr).c_str(), source_str, (exp_params.sequence_start_list[source] + f * steps * skip));
					if (exp_params.parameter<bool>("wrong_backward", "0"))
						sprintf(bFlowF, ("%s%s" + flow_format + "_back.flo").c_str(), (exp_params.output + hfr).c_str(), source_str,
								(exp_params.sequence_start_list[source] + f * steps * skip));
					else
						sprintf(bFlowF, ("%s%s" + flow_format + "_back.flo").c_str(), (exp_params.output + hfr).c_str(), source_str,
								(exp_params.sequence_start_list[source] + f * steps * skip + steps * skip));

					if (!boost::filesystem::exists(fFlowF)) {
						cerr << fFlowF << " does not exist!" << endl;
						return -1;
					}
					if (!boost::filesystem::exists(bFlowF)) {
						cerr << bFlowF << " does not exist!" << endl;
						return -1;
					}

					cout << "Reading " << fFlowF <<  endl;
					cout << "Reading " << bFlowF <<  endl;

					forward_flow[f] = readGTMiddlebury(fFlowF);
					backward_flow[f] = readGTMiddlebury(bFlowF);

					// crop images
					if(exp_params.center.x > 0) {
						// NOTE: ROWRANGE IS INDUCING ERRORS IN ACCUMULATION!!!!!!
						forward_flow[f] = crop(forward_flow[f], exp_params.center, exp_params.extent);
						backward_flow[f] = crop(backward_flow[f], exp_params.center, exp_params.extent);
					}
				}
				Mat lfr_occlusions = Mat::zeros(forward_flow[0].rows, forward_flow[0].cols, CV_8UC1);
				if(gt != NULL && gt_occlusions != NULL) lfr_occlusions = fuseOcclusions(gt, gt_occlusions, 0, Jets);

				// accumulate ground truth flow
				Mat fconf, bconf;
				Mat* acc_flow = new Mat[Jets];
//				accumulateFlow(acc_flow,  &forward_flow[b * Jets], lfr_occlusions, Jets);
				accumulateConsistentBatches(acc_flow, forward_flow, backward_flow, fconf, bconf, NULL, Jets, params.parameter<double>("acc_consistency_threshold_slope"), 0, params.parameter<bool>("acc_discard_inconsistent", "1"), 0);

				/*
				 *  ########################################### Refine flow ###########################################
				 */

//				if(exp_params.parameter<bool>("refine", "1")) {
//					int width = acc_flow->cols;
//					int height = acc_flow->rows;
//
//					ParameterList refine_params(exp_params);
//					setDefaultVariational(refine_params);
//					refine_params.F = 2;
//					refine_params.Jets = 1;
//					refine_params.insert("S", "2", true);
//					refine_params.verbose = "1000";
//					refine_params.insert("layers", "1", true);
//					refine_params.insert("method", "forward", true);
//					refine_params.insert("niter_alter", "1", true);
//					refine_params.insert("niter_outer", "5", true);
//					refine_params.insert("occlusion_reasoning", "0", true);
//					refine_params.insert("framerate_reasoning", "0", true);
//					refine_params.insert("rho_0", "1", true);
//					refine_params.insert("omega_0", "0", true);
//
//					color_image_t** im = new color_image_t*[3];
//					im[0] = color_image_new(width, height);
//					im[1] = color_image_new(width, height);
//					im[2] = color_image_new(width, height);
//

//					/*
//					 * ################### read in image sequence ###################
//					 */
//					for (uint32_t f = 0; f < 2; f++) {
//						char img_file[1024];
//						if (!sintel) {
//							sprintf(img_file, (sequence_path + format).c_str(), thread_params.sequence_start_list[source] + f * Jets * steps * skip);
//						} else {
//							int sintel_frame = thread_params.sequence_start_list[source] / 1000;
//							int hfr_frame = f * steps * skip + (thread_params.sequence_start_list[source] % 1000);
//
//							while (hfr_frame < 0) {
//								sintel_frame--;
//								hfr_frame = 42 + hfr_frame;
//							}
//							while (hfr_frame > 41) {
//								sintel_frame++;
//								hfr_frame = hfr_frame - 42;
//							}
//
//							sprintf(img_file, (sequence_path + format).c_str(), sintel_frame, hfr_frame);
//						}
//						cout << "Reading " << img_file << "..." << endl;
//
//						sequence[f] = imread(string(img_file), CV_LOAD_IMAGE_UNCHANGED);         // load images
//
//						float norm = 1;
//						if (sequence[f].type() == 2 || sequence[f].type() == 18)
//							norm = 1.0f / 255;		// for 16 bit images
//
//						// convert to floating point
//						sequence[f].convertTo(sequence[f], CV_32FC(sequence[f].channels()));
//
//						/*
//						 * DEMOSAICING
//						 */
//						if (thread_params.exists("raw") && thread_params.parameter<bool>("raw")) {
//							Mat tmp = sequence[f].clone();
//							color_image_t* tmp_in = color_image_new(sequence[f].cols, sequence[f].rows);
//							color_image_t* tmp_out = color_image_new(sequence[f].cols, sequence[f].rows);
//
//							switch (thread_params.parameter<int>("raw_demosaicing", "0")) {
//								case 0: // use bilinear demosaicing
//									sequence[f] = Mat::zeros(tmp.rows, tmp.cols, CV_32FC3);
//									//						bayer2rgb(tmp, sequence[f], green_start, blue_start); 	// red green
//									bayer2rgbGR(tmp, sequence[f], red_loc[0], red_loc[1]); // red green
//									break;
//
//								case 1:	// use hamilton adams demosaicing
//									mat2colorImg<float>(sequence[f], tmp_in);
//
//									HamiltonAdamsDemosaic(tmp_out->c1, tmp_in->c1, tmp_in->width, tmp_in->height, red_loc[0], red_loc[1]); // Hamilton-Adams implemented by Pascal Getreuer
//
//									sequence[f] = Mat::zeros(sequence[f].rows, sequence[f].cols, CV_32FC3);
//									colorImg2colorMat<Vec3f>(tmp_out, sequence[f]);
//									break;
//
//								case 2: // use opencv demosaicing
//									tmp.convertTo(tmp, CV_8UC1);
//									sequence[f] = Mat::zeros(tmp.rows, tmp.cols, CV_8UC3);
//
//									int code = CV_BayerBG2RGB;
//									if (red_loc[1] == 0) // y
//										if (red_loc[0] == 0) // x
//											code = CV_BayerBG2RGB;
//										else
//											code = CV_BayerGB2RGB;
//									else if (red_loc[0] == 0) // x
//										code = CV_BayerGR2RGB;
//									else
//										code = CV_BayerRG2RGB;
//
//									cv::cvtColor(tmp, sequence[f], code); // components from second row, second column !!!!!!!!!!!!!!!!!
//									sequence[f].convertTo(sequence[f], CV_32FC(sequence[f].channels()));
//									break;
//							}
//
//							color_image_delete(tmp_in);
//							color_image_delete(tmp_out);
//						} else {
//							// covert to RGB
//							cv::cvtColor(sequence[f], sequence[f], CV_BGR2RGB);
//						}
//
//						/*
//						 * COLOR CORRECTION
//						 */
//						vector<Mat> channels;
//						split(sequence[f], channels);
//						float mid_val = (0.5 * 255.0 / norm);
//						channels[0] = contrast * (channels[0] + red_balance + brightness - mid_val) + mid_val;
//						channels[1] = contrast * (channels[1] + green_balance + brightness - mid_val) + mid_val;
//						channels[2] = contrast * (channels[2] + blue_balance + brightness - mid_val) + mid_val;
//						merge(channels, sequence[f]);
//
//						if (thread_params.parameter<bool>("grayscale", "0"))
//							cvtColor(sequence[f], sequence[f], CV_RGB2GRAY);
//
//						// use only a part of the images
//						if (thread_params.extent.x > 0 || thread_params.extent.y > 0) {
//							sequence[f] = sequence[f].rowRange(Range(thread_params.center.y - thread_params.extent.y / 2, thread_params.center.y + thread_params.extent.y / 2));
//							sequence[f] = sequence[f].colRange(Range(thread_params.center.x - thread_params.extent.x / 2, thread_params.center.x + thread_params.extent.x / 2));
//						}
//
//						// rescale image with gaussian blur to avoid anti-aliasing
//						double img_scale = thread_params.parameter<double>("scale", "1.0");
//						if (img_scale != 1) {
//							GaussianBlur(sequence[f], sequence[f], Size(0, 0), 1 / sqrt(2 * img_scale), 1 / sqrt(2 * img_scale), BORDER_REPLICATE);
//							resize(sequence[f], sequence[f], Size(0, 0), img_scale, img_scale, INTER_LINEAR);
//						}
//
//						// print to file
//						char file[1024];
//						sprintf(file, (acc_folder.str() + "sequence/frame_%s%i.png").c_str(), source_str, thread_params.sequence_start_list[source] - steps * skip + f * skip);
//
//						Mat output_img;
//						if (thread_params.parameter<bool>("16bit", "0")) {
//							sequence[f].convertTo(output_img, CV_16UC(sequence[f].channels()));
//						} else {
//							sequence[f].convertTo(output_img, CV_8UC(sequence[f].channels()), norm);
//						}
//
//						if (thread_params.parameter<bool>("grayscale", "0"))
//							cv::cvtColor(output_img, output_img, CV_GRAY2BGR);	// OpenCV uses BGR
//						else
//							cv::cvtColor(output_img, output_img, CV_RGB2BGR);	// OpenCV uses BGR
//
//						if (thread_params.verbosity(WRITE_FILES)) {
//							imwrite(file, output_img, compression_params);
//						}
//
//						data[f] = color_image_new(sequence[f].cols, sequence[f].rows);
//						if (thread_params.parameter<bool>("grayscale", "0"))
//							mat2colorImg<float>(sequence[f], data[f]);
//						else
//							colorMat2colorImg<Vec3f>(sequence[f], data[f]);
//
//						if (thread_params.verbosity(VER_IN_GT) && (int) f < show_frames) {
//							if (source == show_frames_source) {
//								stringstream ftitle;
//								ftitle << "Frame " << source;
//								namedWindow(ftitle.str());               // Create a window for display.
//								imshow(ftitle.str(), output_img);
//
//								// set the callback function for pixel selection
//								int* cbparams = new int[2];
//								cbparams[0] = f;
//								cbparams[1] = source;
//								setMouseCallback(ftitle.str(), CallBackFunc, cbparams);
//
//								waitKey(0);
//							}
//						}
//					}
//
//					// normalize data terms
//					normalize(data, Jets + 1, thread_params);
//
//					// rescale image with gaussian blur to avoid anti-aliasing
//					double rescale = 1.0f * width / width;
//					Mat tmp_1 = sequence[0].clone(),
//						tmp_2 = sequence[Jets].clone();
//					if (rescale != 1) {
//						GaussianBlur(tmp_1, tmp_1, Size(0, 0), 1 / sqrt(2 * rescale), 1 / sqrt(2 * rescale), BORDER_REPLICATE);
//						GaussianBlur(tmp_2, tmp_2, Size(0, 0), 1 / sqrt(2 * rescale), 1 / sqrt(2 * rescale), BORDER_REPLICATE);
//					}
//					resize(tmp_1, tmp_1, Size(width, height), 0, 0, INTER_LINEAR);
//					resize(tmp_2, tmp_2, Size(width, height), 0, 0, INTER_LINEAR);
//					if (thread_params.parameter<bool>("grayscale", "0")) {
//						mat2colorImg<float>(tmp_2, im[0]);// assumes backward frame for symmetric option
//						mat2colorImg<float>(tmp_1, im[1]);
//						mat2colorImg<float>(tmp_2, im[2]);
//					} else {
//						colorMat2colorImg<Vec3f>(tmp_2, im[0]);// assumes backward frame for symmetric option
//						colorMat2colorImg<Vec3f>(tmp_1, im[1]);
//						colorMat2colorImg<Vec3f>(tmp_2, im[2]);
//					}
//
//					normalize(im, 3, refine_params);
//
//					image_t *wx = image_new(im[0]->width, im[0]->height), *wy = image_new(im[0]->width, im[0]->height);
//					vector<Mat> vu;
//					split(flow[0][Jets - 1], vu);
//					mat2img<double>(vu[1], wx);
//					mat2img<double>(vu[0], wy);
//
////						bool change = true;
////						while(change) {
////							for(int i=0 ; i<wx->height ; i++){
////								for( int j=0 ; j<wx->width ; j++){
////									change = false;
////
////									if(wx->data[i*wx->stride+j] > UNKNOWN_FLOW_THRESH) {
////										wx->data[i*wx->stride+j] = 0;
////										int count = 0;
////										if(wx->data[(i+1)*wx->stride+j] < UNKNOWN_FLOW_THRESH) {
////											wx->data[i*wx->stride+j] += wx->data[(i+1)*wx->stride+j];
////											count++;
////										}
////										if(wx->data[(i-1)*wx->stride+j] < UNKNOWN_FLOW_THRESH) {
////											wx->data[i*wx->stride+j] += wx->data[(i-1)*wx->stride+j];
////											count++;
////										}
////										if(wx->data[i*wx->stride+(j+1)] < UNKNOWN_FLOW_THRESH) {
////											wx->data[i*wx->stride+j] += wx->data[i*wx->stride+(j+1)];
////											count++;
////										}
////										if(wx->data[i*wx->stride+(j-1)] < UNKNOWN_FLOW_THRESH) {
////											wx->data[i*wx->stride+j] += wx->data[i*wx->stride+(j-1)];
////											count++;
////										}
////
////										wx->data[i*wx->stride+j] = wx->data[i*wx->stride+j] / count;
////										change = true;
////									}
////
////									if(wy->data[i*wy->stride+j] > UNKNOWN_FLOW_THRESH) {
////										wy->data[i*wy->stride+j] = 0;
////										int count = 0;
////										if(wy->data[(i+1)*wy->stride+j] < UNKNOWN_FLOW_THRESH) {
////											wy->data[i*wy->stride+j] += wy->data[(i+1)*wy->stride+j];
////											count++;
////										}
////										if(wy->data[(i-1)*wy->stride+j] < UNKNOWN_FLOW_THRESH) {
////											wy->data[i*wy->stride+j] += wy->data[(i-1)*wy->stride+j];
////											count++;
////										}
////										if(wy->data[i*wy->stride+(j+1)] < UNKNOWN_FLOW_THRESH) {
////											wy->data[i*wy->stride+j] += wy->data[i*wy->stride+(j+1)];
////											count++;
////										}
////										if(wy->data[i*wy->stride+(j-1)] < UNKNOWN_FLOW_THRESH) {
////											wy->data[i*wy->stride+j] += wy->data[i*wy->stride+(j-1)];
////											count++;
////										}
////
////										wy->data[i*wy->stride+j] = wy->data[i*wy->stride+j] / count;
////										change = true;
////									}
////								}
////							}
////						}
//
//					// energy minimization
//				//	time(&t_start);
//					Variational_MT minimzer_f;
//					minimzer_f.variational(wx, wy, im, refine_params);
//				//	time(&t_end);
//
//					img2mat<double>(wx, vu[1]);
//					img2mat<double>(wy, vu[0]);
//					merge(vu, flow[0][Jets - 1]);
//
//					color_image_delete(im[0]);
//					color_image_delete(im[1]);
//					color_image_delete(im[2]);
//					delete[] im;
//					image_delete(wx);
//					image_delete(wy);
//				}

				// write final estimation
//				stringstream flowF;
//				flowF << acc_folder.str() << "/" << exp_params.name_list[source].c_str() << source  << "frame_" << exp_params.sequence_start_list[source] + steps * batch_size * b;
				char flowF[200];
				sprintf(flowF, ("%s/%s" + flow_format).c_str(), acc_folder.str().c_str(), source_str, (exp_params.sequence_start_list[source]));


				Mat flow_img = flowColorImg(acc_flow[Jets-1], exp_params.verbosity(VER_CMD));

				if (!flow_img.data) {                 // Check for invalid input
					cout << "No flow for frame " << exp_params.sequence_start_list[source] << std::endl;
					continue;
				}

				imwrite((string(flowF) + "_vis.png"), flow_img);

				writeFlowMiddlebury(acc_flow[Jets-1], (string(flowF) + ".flo"));

				//----------------------------------
				// ########################################### final clean up ###########################################
				//----------------------------------


				if(gt != NULL) 				delete[] gt;
				if(gt_occlusions != NULL) 	delete[] gt_occlusions;
				delete[] forward_flow;
				delete[] backward_flow;
			}
		}
	}

	return 0;
}


