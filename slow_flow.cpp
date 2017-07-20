/*
 * slow_flow.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: Janai
 */
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cmath>
#include <omp.h>
#include <stdio.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "epic_flow_extended/image.h"
#include "epic_flow_extended/io.h"
#include "epic_flow_extended/epic.h"
#include "epic_flow_extended/variational_mt.h"
#include "utils/utils.h"
#include "utils/parameter_list.h"
#include "configuration.h"

using namespace std;
using namespace cv;

enum COMPARISON {GROUNDTRUTH = 0, WARPING = 1};

/* show usage information */
void usage(){
    printf("usage:\n");
    printf("    ./slow_flow [cfg] -overwrite -resume -deep_settings [settings] -threads -fr [select one specific adaptive frame rate] -jet [select one specific high speed flow]\n");
    printf("\n");
}

void setDefault(ParameterList& params) {
    // general
    params.insert("verbose", "0", true);
    params.insert("threads", "1", true);

    params.insert("16bit", "1", true);									// set to 1 if input images 16 bit
    params.insert("raw", "1", true);									// set to 1 if raw images
    params.insert("raw_weight", "1", true);								// weight for raw pixel
    params.insert("raw_demosaicing", "1", true);						// demosaicing method 0: bilinear interp, 1: hamilton adams, 2: opencv
    params.insert("raw_red_loc", "1,0", true);							// location of first red pixel (x,y)

    params.insert("Jets", "1", true);									// number of high speed flow estimates
    params.insert("adaptive", "1", true);								// choose two frame rates according to 0.99 quantile
    params.insert("max_fps", "200", true);								// frame rate of input sequence
    params.insert("ref_fps", "20", true);								// final frame rate


    params.insert("scale", "1.0f", true);								// scaling factor for input images
    params.insert("sigma", "0.0f", true);								// presmoothing
    params.insert("deep_matching", "1", true);							// set to 1 to use deep matching
    params.insert("dm_scale", "1.0f", true);							// scaling factor for deep matching

    params.insert("slow_flow_method", "symmetric", true);				// symmetric: symmetric window, forward: forward window
    params.insert("slow_flow_S", "2", true);							// number of frames in window

    // energy function
    params.insert("slow_flow_dataterm", "1", true);						// 0: unnormalized, 1: normalized
    params.insert("slow_flow_smoothing", "1", true);					// 0: \phi(u_dx) + \phi(u_dy), 1: \phi(u_dx + u_symdy) + \phi(u_dy + u_symdx), 2: \phi(u_dx + u_dy)
    params.insert("slow_flow_alpha", "4.0f", true);						// flow smoothness weight
    params.insert("slow_flow_gamma", "6.0f", true);						// gradient constancy assumption
    params.insert("slow_flow_delta", "1.0f", true);						// color constancy assumption weight

    params.insert("slow_flow_rho_0", "1", true);						// weight for successive data terms
    params.insert("slow_flow_rho_1", "1", true);						// weight for successive data terms
    params.insert("slow_flow_omega_0", "0", true);						// weight for reference data terms
    params.insert("slow_flow_omega_1", "2", true);						// weight for reference data terms

    // image pyramid
    params.insert("slow_flow_layers", "1", true);						// number of pyramid layers
    params.insert("slow_flow_p_scale", "0.9f", true);					// scaling factor for pyramid

    // optimization
    params.insert("slow_flow_niter_alter", "10", true);					// number of alternations
    params.insert("slow_flow_niter_graphc", "10", true);				// number of iterations for graph cut expansion algorithm
    params.insert("slow_flow_niter_outer", "10", true);					// number of outer fixed point iterations
    params.insert("slow_flow_thres_outer", "1e-5", true);				// threshold for du and dv to stop the optimization
    params.insert("slow_flow_niter_inner", "1", true);					// number of inner fixed point iterations
    params.insert("slow_flow_thres_inner", "1e-5", true);				// threshold for du and dv to stop the optimization
    params.insert("slow_flow_niter_solver", "30", true);				// number of solver iterations
    params.insert("slow_flow_sor_omega", "1.9f", true);					// omega parameter of sor method

    // occlusion reasoning
    params.insert("slow_flow_occlusion_reasoning", "1", true);			// set to 1 to enable occlusion reasoning
    params.insert("slow_flow_occlusion_penalty", "0.1", true);			// preference of backwards occlusion (using the forward data terms)
    params.insert("slow_flow_occlusion_alpha", "0.1", true);			// occlusion smoothness weight
    params.insert("slow_flow_output_occlusions", "1", true);			// set to 1 to output occlusion estimate

    // regularization
    params.insert("slow_flow_robust_color", "1", true);					// 0: quadratic, 1: modified_l1_norm, 2: lorentzian, 3: trunc mod l1, 4: Geman McClure
    params.insert("slow_flow_robust_color_eps", "0.001", true);			// epsilon of robust function
    params.insert("slow_flow_robust_color_truncation", "0.5", true);	// truncation threshold (trun_mod_l1)
    params.insert("slow_flow_robust_reg", "1", true);					// 0: quadratic, 1: modified_l1_norm, 2: lorentzian, 3: trunc mod l1, 4: Geman McClure
    params.insert("slow_flow_robust_reg_eps", "0.001", true);			// epsilon of robust function
    params.insert("slow_flow_robust_reg_truncation", "0.5", true);		// truncation threshold (trun_mod_l1)
}

inline bool insideImg(double x, double y, int width, int height) {
	return (y >= 0 && y < height && x >= 0 && x < width);
}

int main(int argc, char **argv){
    if( argc < 2){
        if(argc>1) fprintf(stderr,"Error, not enough arguments\n");
        usage();
        exit(1);
    }

    int n_thread = 1;

    // read optional arguments
    string sequence_path = "", output = "", format = "";
    uint32_t start = 0;

    ParameterList params;
    setDefault(params);	// set default parameters
    // read in ParameterList
    if( argc > 1 && argv[1][0] != '-' && access( argv[1], F_OK ) != -1) {
    	params.read(argv[1]);
    } else {
    	cerr << "Couldn't find " << argv[1] << "!" << endl;
    	return -1;
    }

	#define isarg(key)  !strcmp(a,key)
    bool overwrite_output = false;
    bool resume_frame = false;
	string input_path = "";
	string output_path = "";
	string deep_settings = "";
	double max_flow_scale = 3.0;

    int selected_jet = -1;
    int selected_fr = -1;
	int current_arg = 1;
	while(current_arg < argc ){
		const char* a = argv[current_arg++];
		if(a[0] != '-') {
			continue;
		}

		if( isarg("-h") || isarg("-help") )
			usage();
		else if( isarg("-overwrite") )
			overwrite_output = true;
		else if( isarg("-resume") )
			resume_frame = true;
		else if( isarg("-deep_settings") )
			deep_settings = string(argv[current_arg++]);
		else if( isarg("-threads") )
			params.insert("threads", argv[current_arg++], true);
		else if( isarg("-fr") )
			selected_fr = atoi(argv[current_arg++]);
		else if( isarg("-jet") ) {
			selected_jet = atoi(argv[current_arg++]);
			resume_frame = true;
		}else{
			fprintf(stderr, "unknown argument %s", a);
			usage();
		}
	}

	bool enable_dm = params.parameter<bool>("deep_matching");

	// to restrict deep call -deep_settings '-ngh_rad <max_flow>'
	float max_flow = 50;
	if(params.exists("max_flow"))
		max_flow = max(5.0f, params.parameter<float>("max_flow"));

    float scale = params.parameter<float>("scale","1.0");
	n_thread = params.parameter<int>("threads");

	start = params.sequence_start;

	// decompose sequence in batches
	int steps = params.parameter<int>("slow_flow_S") - 1;
    int ref = steps;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	int max_fps = params.parameter<int>("max_fps", "1");									// sequence fps
	int jet_fps = max_fps;
	if(params.exists("jet_fps")) jet_fps = params.parameter<int>("jet_fps");				// jet estimation fps
	int skip = (1.0f * max_fps) / jet_fps;													// number of frames to skip for jet fps

	cout << skip << endl;

	// split sequence path into path an format
	bool sintel = params.parameter<bool>("sintel", "0");		// specific file names (we would like to be able to distinguish frame number from 24 fps and 1008 fps)
	bool subframes = params.parameter<bool>("subframes", "0");	// are subframes specified

	int start_format = (params.file.find_last_of('/') + 1);
	int end_format = params.file.length() - start_format;

	sequence_path = params.file.substr(0,start_format);
	format = params.file.substr(start_format, end_format);
	if(sequence_path[sequence_path.length() - 1] != '/') sequence_path = sequence_path + "/";

	params.file = sequence_path;
	params.insert("format", format, true);

	if(sequence_path.empty() || params.output.empty())
		return -1;

	int len_format = format.find_last_of('.');
	string format_flow = format.substr(0,len_format);

	if(sintel && !subframes) {
		start = start * 1000;
		for(uint32_t i = 0; i < params.sequence_start_list.size(); i++) {
			params.sequence_start_list[i] = params.sequence_start_list[i] * 1000; 	//
		}
	}

	params.sequence_start = start;

	// MAKE SURE FOLDER IS NOT OVERWRITTEN
	if(!resume_frame && !overwrite_output) {
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

	epic_params_t epic_params;
	epic_params_default(&epic_params);
	epic_params.pref_nn= 25;
	epic_params.nn= 160;
	epic_params.coef_kernel = 1.1f;

	/*
	 * ################## read in quantil
	 */
    bool adaptive = false;
    double quantil = 1.0;
	double hfr_quantil = 2.0;
	int hfr_rate = 1;
	int lfr_rate = 4;

    string adfr = SOURCE_PATH + "/adaptiveFR.dat";
	if (access(adfr.c_str(), F_OK) != -1) {
		char line[500];
		fstream adativef;
		adativef.open(adfr.c_str());

		while(adativef.getline(line, 500)) {
			char* split = strtok(line, "\n");          // get rid of '\n' at the end
			split = strtok(split, "\t");
			char* val = strtok(NULL, "\t");

			if(strcmp(split, "opt_hfr_quantil") == 0) {
				hfr_quantil = atof(val);
			}
			if(strcmp(split, "opt_lfr_rate") == 0) {
				lfr_rate = atof(val);
			}
		}
		adaptive = params.parameter<bool>("adaptive", "0");

		adativef.close();
	}

	double orig_max_flow = 0;
	string qfstr = (sequence_path+"/quantil.dat");
	if (!params.exists("max_flow") && access(qfstr.c_str(), F_OK) != -1) {
		char line[500];
		fstream quantilf;
		quantilf.open(qfstr.c_str());
		quantilf.getline(line, 500);
		quantil = atof(line);

		// use maximum or quantil
		if(quantilf.getline(line, 500))
			orig_max_flow = max_flow_scale * atof(line);
		else
			orig_max_flow = max_flow_scale * quantil;

		// compute frame rate
		if(adaptive) {
			int keyframes = params.parameter<float>("max_fps") / params.parameter<float>("ref_fps");

			if(keyframes == 0) {
				// exact rates
				hfr_rate = hfr_quantil / quantil;
				hfr_rate = max(1.0, round(hfr_rate));		// rounded and minimum 1
				lfr_rate = hfr_rate * lfr_rate;				// too small quantils will have the same hfr and lfr this way!

				lfr_rate = hfr_rate * lfr_rate;

				// make sure we have the same keyframes
				double m = round(lfr_rate / hfr_rate);
				lfr_rate = hfr_rate * m;
				cout << hfr_rate << " " << lfr_rate << endl;
			} else {
				// with keyframes
				hfr_rate = max(1.0, round(hfr_quantil / quantil));	// rounded and minimum 1
				while(hfr_rate < keyframes && keyframes % (hfr_rate * steps) != 0)
					hfr_rate++;
				cout << "hfr_rate " << hfr_rate << endl;

				lfr_rate = min(keyframes, hfr_rate * lfr_rate);
				while((lfr_rate * steps < keyframes && (keyframes % (lfr_rate * steps) != 0 || (keyframes % (lfr_rate * steps) == 0 && (lfr_rate * steps) % (hfr_rate * steps) != 0))) ||
						(lfr_rate * steps >= keyframes && (lfr_rate * steps) % (hfr_rate * steps) != 0))
					lfr_rate++;
				lfr_rate = min(keyframes / steps, lfr_rate);

				cout << "lfr_rate " << lfr_rate << endl;
			}
		} else {
			// set maximum flow according to quantil
			max_flow = max(5.0, orig_max_flow * scale * ref * skip);	// twice to make sure its big enough and at least 5 pixel
		}
	} else
		adaptive = false;

	int start_fr = 0;
	int end_fr = (adaptive + 1);
	if(selected_fr >= 0) {
		start_fr = selected_fr;
		end_fr = selected_fr + 1;
	}

	string orig_deep_settings = deep_settings;
	for(int adFR = start_fr; adFR < end_fr; adFR++) {
		ParameterList adaptCfg(params);

		deep_settings = orig_deep_settings;
		if(adaptive) {
			stringstream jfps;
			if(adFR == 0) {
				adaptCfg.output += "high_fr/";

				// set frame rate
				jfps << max_fps / hfr_rate;
				adaptCfg.insert("jet_fps", jfps.str(), true);				// jet estimation fps
				skip = hfr_rate;

				// compute max flow
				max_flow = max(5.0, orig_max_flow * scale * ref * hfr_rate);
			} else {
				adaptCfg.output += "low_fr/";

				// set frame rate
				jfps << max_fps / lfr_rate;
				adaptCfg.insert("jet_fps", jfps.str(), true);				// jet estimation fps
				skip = lfr_rate;

				// compute max flow
				max_flow = max(5.0, orig_max_flow * scale * ref * lfr_rate);
			}
		}

		// SMALLER RESOLUTION FOR DEEP MATCHING
	    double dm_scale = params.parameter<float>("dm_scale","1.0");
		if(enable_dm && max_flow > 150) {
			dm_scale = 0.5 * dm_scale;

			max_flow = max(5.0, 0.5 * max_flow);
		}

		/*
		 * frames = 1 						for reference frame
		 * 			+ steps 				window before first frame
		 * 			+ (Jets - 1) * steps 	window for each jet
		 * 			+ steps					window after last frame
		 * 			+ steps					additional window for backward flow
		 */
		int frames = 1 + (adaptCfg.Jets + 2) * steps;

		// only read necessary frames
		uint32_t start_f = 0;
		uint32_t end_f = frames;
		uint32_t start_j = 0;
		uint32_t end_j = adaptCfg.Jets;
		if(resume_frame && selected_jet >= 0) {
			start_f = selected_jet * steps;
			end_f = min(frames, 1 + (selected_jet + 3) * steps);

			start_j = selected_jet;
			end_j = min((int) adaptCfg.Jets, selected_jet + 1);
		}

		if(start_f > end_f) continue;

	    // create results folder
	    boost::filesystem::create_directories(adaptCfg.output);                             // ParameterList result folder
	    boost::filesystem::create_directories(adaptCfg.output+"/sequence/");                // ParameterList result folder
	    if(!adaptCfg.file_gt.empty()) boost::filesystem::create_directories(adaptCfg.output+"/gt/");              		  // ParameterList result folder

	    int width = 0;
	    int height = 0;

		/*
		 * ################### read in image sequence ###################
		 */
		vector<int> red_loc = adaptCfg.splitParameter<int>("raw_red_loc","0,0");

		char** img_files = new char*[frames];
		color_image_t **un_seq = new color_image_t*[frames];
		color_image_t **un_seq_back = new color_image_t*[frames];
		color_image_t **seq = new color_image_t*[frames];
		color_image_t **seq_back = new color_image_t*[frames];

		for(uint32_t f = start_f; f < end_f; f++) {
			char img_file[200];

			if(!sintel) {
				sprintf(img_file, (sequence_path+format).c_str(), start - ref * skip + f * skip);
			} else {
				int sintel_frame = start / 1000;
				int hfr_frame = f * skip - ref * skip + (start % 1000);

				while(hfr_frame < 0) {
					sintel_frame--;
					hfr_frame = 42 + hfr_frame;
				}
				while(hfr_frame > 41) {
					sintel_frame++;
					hfr_frame = hfr_frame - 42;
				}

				sprintf(img_file, (sequence_path+format).c_str(), sintel_frame, hfr_frame);
			}

			cout << "Reading " << img_file << "..." << endl;

			Mat img = imread(string(img_file), CV_LOAD_IMAGE_UNCHANGED);         // load images

			float norm = 1;
			if(img.type() == 2 || img.type() == 18)
				norm = 1.0f/255;		// for 16 bit images

			// convert to floating point
			img.convertTo(img, CV_32FC(img.channels()));

			/*
			 * DEMOSAICING
			 */
			if(adaptCfg.exists("raw") && adaptCfg.parameter<bool>("raw")) {
				Mat tmp = img.clone();
				color_image_t* tmp_in = color_image_new(img.cols, img.rows);
				color_image_t* tmp_out = color_image_new(img.cols, img.rows);

				switch(adaptCfg.parameter<int>("raw_demosaicing", "0")) {
					case 0: // use bilinear demosaicing
							img = Mat::zeros(tmp.rows, tmp.cols, CV_32FC3);
							bayer2rgbGR(tmp, img, red_loc[0], red_loc[1]); // red green
							break;

					case 1:	// use hamilton adams demosaicing
							mat2colorImg<float>(img, tmp_in);

							HADemosaicing(tmp_out->c1, tmp_in->c1, tmp_in->width, tmp_in->height, red_loc[0], red_loc[1]); // Hamilton-Adams implemented by Pascal Getreuer

							img = Mat::zeros(img.rows, img.cols, CV_32FC3);
							colorImg2colorMat<Vec3f>(tmp_out, img);
							break;

					case 2: // use opencv demosaicing
							tmp.convertTo(tmp, CV_8UC1);
							img = Mat::zeros(tmp.rows, tmp.cols, CV_8UC3);

							int code = CV_BayerBG2RGB;
							if(red_loc[1] == 0) // y
								if(red_loc[0] == 0) // x
									code = CV_BayerBG2RGB;
								else
									code = CV_BayerGB2RGB;
							else
								if(red_loc[0] == 0) // x
									code = CV_BayerGR2RGB;
								else
									code = CV_BayerRG2RGB;

							cv::cvtColor(tmp, img, code); // components from second row, second column !!!!!!!!!!!!!!!!!
							img.convertTo(img, CV_32FC(img.channels()));
							break;
				}

				color_image_delete(tmp_in);
				color_image_delete(tmp_out);
			} else {
				// covert to RGB
				cv::cvtColor(img, img, CV_BGR2RGB);
			}


			if(!adaptCfg.exists("raw") || adaptCfg.parameter<float>("raw_weight", "1.0") == 1.0) {
				// use only a part of the images
				if(adaptCfg.extent.x > 0 || adaptCfg.extent.y > 0) {
					img = img.rowRange(Range(adaptCfg.center.y - adaptCfg.extent.y/2,adaptCfg.center.y + adaptCfg.extent.y/2));
					img = img.colRange(Range(adaptCfg.center.x - adaptCfg.extent.x/2,adaptCfg.center.x + adaptCfg.extent.x/2));
				}

				// rescale image with gaussian blur to avoid anti-aliasing
				if(scale != 1) {
					GaussianBlur(img, img, Size(),1/sqrt(2*scale),1/sqrt(2*scale),BORDER_REPLICATE);
					resize(img, img, Size(0,0), scale, scale, INTER_LINEAR);
				}
			}

			// print to file
			img_files[f] = new char[500];
			sprintf(img_files[f], (adaptCfg.output+"sequence/frame_%i.png").c_str(), start - ref * skip + f * skip);

			Mat output_img;
			if(adaptCfg.verbosity(WRITE_FILES)) {
				if(adaptCfg.parameter<bool>("16bit", "0")) {
					img.convertTo(output_img, CV_16UC(img.channels()));
				} else {
					img.convertTo(output_img, CV_8UC(img.channels()), norm);
				}
				cv::cvtColor(output_img, output_img, CV_RGB2BGR);	// OpenCV uses BGR
				imwrite(img_files[f], output_img, compression_params);
			}

			width = img.cols;
			height = img.rows;

			// copy data
			seq[f] = color_image_new(width, height);
			if(img.channels() == 1) {
				mat2colorImg<float>(img, seq[f]);
			} else
				colorMat2colorImg<Vec3f>(img, seq[f]);

			// resize and copy data for deep match
			if(dm_scale != 1) {
				GaussianBlur(img, img,Size(),1/sqrt(2*dm_scale),1/sqrt(2*dm_scale),BORDER_REPLICATE);
				resize(img, img, Size(0,0), dm_scale, dm_scale, INTER_LINEAR);
			}

			sprintf(img_files[f], (adaptCfg.output+"sequence/frame_epic_%i.png").c_str(), start - ref * skip + f * skip);

			img.convertTo(img, CV_8UC(img.channels()), norm);
			if(f % steps == 0) {
				cv::cvtColor(img, output_img, CV_RGB2BGR);	// OpenCV uses BGR
				imwrite(img_files[f], output_img, compression_params);
			}

			un_seq[f] = color_image_new(width*dm_scale, height*dm_scale);
			if(img.channels() == 1) {
				mat2colorImg<uchar>(img, un_seq[f]);
			} else
				colorMat2colorImg<Vec3b>(img, un_seq[f]);

			un_seq_back[frames - 1 - f] = un_seq[f];
			seq_back[frames - 1 - f] = seq[f];
		}

		/*
		 * DEMOSAICING AND CHANNEL WEIGHTING
		 */
		color_image_t* channel_weights = color_image_new(seq[start_f]->width, seq[start_f]->height);
		fill_n(channel_weights->c1, 3*channel_weights->height*channel_weights->stride, 1.0);												// set channel weights to 1
		if(adaptCfg.exists("raw") && adaptCfg.parameter<bool>("raw"))
			rawWeighting(channel_weights, red_loc[0], red_loc[1], adaptCfg.parameter<float>("raw_weight", "1.0"));


		/*
		 * ################### read in ground truth ###################
		 */
		Mat* gt = new Mat[adaptCfg.Jets];
		for(u_int32_t j = start_j; j < end_j; j++) {
			char path[200];

			if(!sintel)
				sprintf(path, adaptCfg.file_gt.c_str(), start + j*steps);
			else {
				int sintel_frame = start / 1000;
				int hfr_frame = j*steps + (start % 1000);

				while(hfr_frame < 0) {
					sintel_frame--;
					hfr_frame = 42 + hfr_frame;
				}
				while(hfr_frame > 41) {
					sintel_frame++;
					hfr_frame = hfr_frame - 42;
				}

				sprintf(path, adaptCfg.file_gt.c_str(), sintel_frame, hfr_frame);
			}

			cout << path << endl;

			if(access( path, F_OK ) != -1) {
				gt[j] = readGTMiddlebury(string(path));

				// crop images
				if(adaptCfg.center.x > 0) {
					// NOTE: ROWRANGE IS INDUCING ERRORS IN ACCUMULATION!!!!!!
					gt[j] = crop(gt[j], adaptCfg.center, adaptCfg.extent);
				}

				// rescale image
				resize(gt[j], gt[j], Size(0, 0), scale, scale, INTER_NEAREST); // LINEAR PROBLEMATIC AT MOTION DISCONTINOUTIES
				gt[j] *= scale;

				Mat img = flowColorImg(gt[j], adaptCfg.verbosity(VER_CMD));
				if(! img.data) {                 // Check for invalid input
					cout <<  "No gt flow for frame " << endl ;
					continue;
				}

				// write flow and flow image to file
				if(!gt[j].empty() && !adaptCfg.output.empty()) {
					char gtF[200];
					sprintf(gtF, "%s/gt/flow_%05i.png", adaptCfg.output.c_str(), adaptCfg.sequence_start + j*steps);

					imwrite(gtF, img);
				}

				char gtF[200];
				sprintf(gtF, "%s/gt/flow_%05i.flo", adaptCfg.output.c_str(), adaptCfg.sequence_start + j*steps);
				writeFlowMiddlebury(gt[j], gtF);

				// DEBUG: show groundtruth flow
				if(adaptCfg.verbosity(VER_IN_GT)) {
					stringstream title;
					title << "GT flow of frame " << adaptCfg.sequence_start + j*steps;
					namedWindow( title.str(), WINDOW_FREERATIO );               // Create a window for display.
					imshow( title.str(), img );                            // Show our image inside it.
					waitKey(0);
				}
			}
		}

		// normalize intensities
		normalize(&seq[start_f], end_f - start_f, adaptCfg);

		boost::filesystem::create_directories(adaptCfg.output+"tmp/");                				// temporary folder for edges and deep_matches
		if(adaptCfg.parameter<bool>("slow_flow_occlusion_reasoning", "0"))
			boost::filesystem::create_directories(adaptCfg.output+"occlusion/");                	// folder for occlusion output

		/*
		 *  write infos to file
		 */
		adaptCfg.print();

		ofstream infos;
		infos.open((adaptCfg.output + "config.cfg").c_str());
		infos << "# SlowFlow variational estimation\n";
		infos << adaptCfg;
		infos.close();

		// write stats
		stringstream results;
		results << "frame\ttime\n\n";
		int avg_time = 0;
		int counter = 0;

		if(enable_dm && max_flow < 300) {
			stringstream mfstr;
			mfstr << ceil(max_flow);
			deep_settings = " -ngh_rad " + mfstr.str();

			cout << deep_settings << endl;
			cout << "Max flow: " << max_flow << endl;
		} else
			deep_settings = "";

		#pragma omp parallel for num_threads(n_thread) shared(seq, seq_back, gt)
		for(uint32_t j = start_j; j < end_j; j++) {
			ParameterList thread_params(adaptCfg);

			int f = j*steps;

			char curr_f [33];
			sprintf(curr_f, "%d_forward", start + j*steps) ;
			thread_params.insert("current_frame", curr_f, true);

			/*
			 * indices				0 1 2  3  4  5  6  7	8	9			F-4	F-3	F-2 F-1 F   F+1 F+2 F+3 F+4 F+5
			 * forward (S = 3) : 	0 1 2 (3->4->5->6) 7 	8	9	...
			 * backward (S = 3): 					   				... 	9	8	7  (6-> 5-> 4-> 3)  2   1   0
			 */
			color_image_t **un_im = &un_seq[f];
			color_image_t **un_im_back = &un_seq_back[frames - 1 - f - 3 * steps];
			color_image_t **im = &seq[f];
			color_image_t **im_back = &seq_back[frames - 1 - f - 3 * steps];	// -1 because zero index and -steps because different reference frame

			// prepare variables
			image_t *wx, *wy;

			char edges_f[1000], edges_b[1000], edges_cmd[1000], match_f[1000], match_b[1000], match_cmd[1000];
			time_t t_start, t_stop;
			time_t pp_start, pp_stop;
			time_t ep_start, ep_stop;
			int t_preprocessing = 0;

			/*
			 * Compute EDGES AND MATCHES!!!!!
			 */

			sprintf(edges_f, (adaptCfg.output+"tmp/edges_%i.dat").c_str(), adaptCfg.sequence_start + f * skip);
			sprintf(edges_b, (adaptCfg.output+"tmp/edges_%i.dat").c_str(), adaptCfg.sequence_start + f * skip + ref * skip);
			sprintf(match_f, (adaptCfg.output+"tmp/matches_%i_%i.dat").c_str(), adaptCfg.sequence_start + f * skip, adaptCfg.sequence_start + f * skip + ref * skip);
			sprintf(match_b, (adaptCfg.output+"tmp/matches_%i_%i.dat").c_str(), adaptCfg.sequence_start + f * skip + ref * skip, adaptCfg.sequence_start + f * skip);

			if(enable_dm) {
				if(!resume_frame || (resume_frame && access( edges_f, F_OK ) == -1)) {
					cout << "Computing edges ..." << endl;
					sprintf(edges_cmd, "matlab -nodesktop -nojvm -r \"addpath(\'%s/matlab/\'); detect_edges(\'%s\',\'%s\'); exit\"", SOURCE_PATH.c_str(), img_files[j * steps + ref], edges_f);

					time(&pp_start);
					system(edges_cmd);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);
				}

				if(!resume_frame || (resume_frame && access( edges_b, F_OK ) == -1)) {
					cout << "Computing edges ..." << endl;
					sprintf(edges_cmd, "matlab -nodesktop -nojvm -r \"addpath(\'%s/matlab/\'); detect_edges(\'%s\',\'%s\'); exit\"", SOURCE_PATH.c_str(), img_files[j * steps + 2*ref], edges_b);

					time(&pp_start);
					system(edges_cmd);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);
				}


				if(!resume_frame || (resume_frame && access( match_f, F_OK ) == -1)) {
					cout << "Computing matches between " << adaptCfg.sequence_start + f * skip << " and " << adaptCfg.sequence_start + f * skip + ref * skip << " ..." << endl;
					sprintf(match_cmd, "%s/deepmatching \'%s\' \'%s\' -png_settings %s -out \'%s\'", DEEPMATCHING_PATH.c_str(), img_files[j * steps + ref], img_files[j * steps + 2*ref], deep_settings.c_str(), match_f);

					time(&pp_start);
					system(match_cmd);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);
				}

				if(!resume_frame || (resume_frame && access( match_b, F_OK ) == -1)) {
					cout << "Computing matches between " << adaptCfg.sequence_start + f * skip + ref * skip << " and " << adaptCfg.sequence_start + f * skip<< " ..." << endl;
					sprintf(match_cmd, "%s/deepmatching \'%s\' \'%s\' -png_settings %s -out \'%s\'", DEEPMATCHING_PATH.c_str(), img_files[j * steps + 2*ref], img_files[j * steps + ref], deep_settings.c_str(), match_b);

					time(&pp_start);
					system(match_cmd);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);
				}
			}

			char forward_flow_file[200];
			if(!sintel)
				sprintf(forward_flow_file, (adaptCfg.output + format_flow + ".flo").c_str(), start + f * skip);
			else
				sprintf(forward_flow_file, (adaptCfg.output + format_flow + ".flo").c_str(), start + f * skip, 0);

			// skip finished frames
			if(!resume_frame || (resume_frame && access( forward_flow_file, F_OK ) == -1)) {
				/*
				 * ################### extract edges and get matches ###################
				 */

				Mat floImg;
				// use deep matching instead of pyramid
				if(enable_dm) {
					wx = image_new(im[ref]->width*dm_scale, im[ref]->height*dm_scale);
					wy = image_new(im[ref]->width*dm_scale, im[ref]->height*dm_scale);
					image_erase(wx);
					image_erase(wy);

					// matches to target frame
					float_image forward_edges = read_edges(edges_f, un_im[ref]->width, un_im[ref]->height);
					time(&pp_start);
					float_image forward_matches = read_matches(match_f);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);

					color_image_t *imlab = rgb_to_lab(un_im[ref]);

					// initilize with deep matches
					cout << "Epic interpolation of forward flow ..." << endl;
					time(&ep_start);
					epic(wx, wy, imlab, &forward_matches, &forward_edges, &epic_params, 1);
					time(&ep_stop);

					// rescale flow
					float fx = im[ref]->width / wx->width;
					float fy = im[ref]->height / wx->height;

					if(fx != 1) {
						Mat tmpx(wx->height, wx->width, CV_32FC1);
						Mat tmpy(wy->height, wy->width, CV_32FC1);
						img2mat<float>(wx, tmpx);
						img2mat<float>(wy, tmpy);

						resize(tmpx, tmpx, Size(im[ref]->width, im[ref]->height), 0, 0, INTER_LINEAR);
						resize(tmpy, tmpy, Size(im[ref]->width, im[ref]->height), 0, 0, INTER_LINEAR);

						image_delete(wx); image_delete(wy);
						wx = image_new(im[ref]->width, im[ref]->height);
						wy = image_new(im[ref]->width, im[ref]->height);
						mat2img<float>(tmpx, wx);
						mat2img<float>(tmpy, wy);
					}

					image_mul_scalar(wx, fx / steps);									// scale flow vectors
					image_mul_scalar(wy, fy / steps);									// scale flow vectors

					floImg = flowColorImg(wx, wy, adaptCfg.verbosity(VER_CMD));
					// write flow image to file
					if(adaptCfg.verbosity(WRITE_FILES)) {
						if(!floImg.data) {                 // Check for invalid input
							cout <<  "No forward flow for frame " << start + f * skip << std::endl ;
						} else {
							if(!adaptCfg.output.empty()) {
								stringstream flowF;
								flowF << adaptCfg.output <<  "tmp/frame_" << start + f * skip << "_INIT.png";

								imwrite((flowF.str()), floImg);
							}
						}
					}

					color_image_delete(imlab);

					free(forward_matches.pixels);
					free(forward_edges.pixels);
				} else {
					wx = image_new(im[ref]->width, im[ref]->height);
					wy = image_new(im[ref]->width, im[ref]->height);
					image_erase(wx);
					image_erase(wy);
				}

				/*
				 * ############ forward flow ##################
				 */
				cout << "Forward flow estimation ..." << endl;
				Variational_MT minimzer_f;
				minimzer_f.setChannelWeights(channel_weights);

				if(thread_params.verbosity(WRITE_FILES) && thread_params.parameter<bool>("slow_flow_output_occlusions","0")) {
					if(thread_params.parameter<bool>("slow_flow_occlusion_reasoning","0")) {
						stringstream occF;
						occF << adaptCfg.output << "tmp/frame_" << start + f * skip << "_";
						thread_params.insert("slow_flow_occlusions_output", occF.str(), true);
					}
				}

				// energy minimization
				time(&t_start);
				minimzer_f.variational(wx, wy, im, thread_params);
				time(&t_stop);

				// output the occlusions
				if(thread_params.parameter<bool>("slow_flow_output_occlusions","0")) {
					image_t* occlusions = minimzer_f.getOcclusions();
					Mat occ_mat(im[ref]->height, im[ref]->width, CV_32FC1);
					img2mat<float>(occlusions, occ_mat);
					occ_mat = 0.5 * (occ_mat + 1);
					occ_mat.convertTo(occ_mat, CV_8UC1, 255);
					stringstream occF;
					occF << adaptCfg.output << "/occlusion/frame_" << start + f * skip << ".pbm";

					vector<int> compression_params_occ;
					compression_params_occ.push_back(CV_IMWRITE_PXM_BINARY);
					compression_params_occ.push_back(1);						// store as binary image
					imwrite(occF.str(), occ_mat, compression_params_occ);
				}

				// write output file
				image_mul_scalar(wx, steps);		// scale flow
				image_mul_scalar(wy, steps);		// scale flow

				writeFlowFile(forward_flow_file, wx, wy);

				floImg = flowColorImg(wx, wy, adaptCfg.verbosity(VER_CMD));

				// write flow image to file
				if(!floImg.data) {                 // Check for invalid input
					cout <<  "No forward flow for frame " << start + f * skip << std::endl ;
				} else {
					if(!adaptCfg.output.empty()) {
						stringstream flowF;
						flowF << adaptCfg.output <<  "frame_" << start + f * skip << ".png";

						imwrite((flowF.str()), floImg);
					}
				}

				int time = (int) difftime(t_stop,t_start) + (int) difftime(ep_stop,ep_start) + t_preprocessing;
				// store results
				#pragma omp critical (sum)
				{
					avg_time += time;
					counter++;

					// add epe for this frame
					results << f * skip << "\t " << time << "\n";
				}

				// clean up
				image_delete(wx);
				image_delete(wy);

				cout << "Forward flow from frame " << start + f << " to " << start + f * skip + steps * skip << " finished! (Computation took " << time << " s)" << endl;
			} else
				cout << "Forward flow from frame " << start + f << " to " << start + f * skip + steps * skip << " already exist!" << endl;

			/*
			 * ############ backward flow ##################
			 */
			thread_params = ParameterList(adaptCfg);

			char backward_flow_file[200];
			if(!sintel)
				sprintf(backward_flow_file, (adaptCfg.output + format_flow + "_back.flo").c_str(), start + f * skip + steps * skip);
			else
				sprintf(backward_flow_file, (adaptCfg.output + format_flow + "_back.flo").c_str(), start + f * skip + steps * skip, 0);

			// skip finished frames
			if(!resume_frame || (resume_frame && access( backward_flow_file, F_OK ) == -1)) {
				t_preprocessing = 0;

				// use deep matching instead of pyramid
				if(enable_dm) {
					wx = image_new(im[ref]->width*dm_scale, im[ref]->height*dm_scale);
					wy = image_new(im[ref]->width*dm_scale, im[ref]->height*dm_scale);
					image_erase(wx);
					image_erase(wy);

					float_image backward_edges = read_edges(edges_b, un_im_back[ref]->width, un_im_back[ref]->height);
					time(&pp_start);
					float_image backward_matches = read_matches(match_b);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);

					color_image_t *imlab = rgb_to_lab(un_im_back[ref]);

					// initilize with deep matches
					cout << "Epic interpolation of backward flow ..." << endl;
					time(&ep_start);
					epic(wx, wy, imlab, &backward_matches, &backward_edges, &epic_params, 1);
					time(&ep_stop);

					// rescale flow
					float fx = im[ref]->width / wx->width;
					float fy = im[ref]->height / wx->height;

					if(fx != 1) {
						Mat tmpx(wx->height, wx->width, CV_32FC1);
						Mat tmpy(wy->height, wy->width, CV_32FC1);
						img2mat<float>(wx, tmpx);
						img2mat<float>(wy, tmpy);

						resize(tmpx, tmpx, Size(im[ref]->width, im[ref]->height), 0, 0, INTER_LINEAR);
						resize(tmpy, tmpy, Size(im[ref]->width, im[ref]->height), 0, 0, INTER_LINEAR);

						image_delete(wx); image_delete(wy);
						wx = image_new(im[ref]->width, im[ref]->height);
						wy = image_new(im[ref]->width, im[ref]->height);
						mat2img<float>(tmpx, wx);
						mat2img<float>(tmpy, wy);
					}

					image_mul_scalar(wx, fx / steps);									// scale flow vectors
					image_mul_scalar(wy, fy / steps);									// scale flow vectors

					color_image_delete(imlab);

					free(backward_matches.pixels);
					free(backward_edges.pixels);
				} else {
					wx = image_new(im[ref]->width, im[ref]->height);
					wy = image_new(im[ref]->width, im[ref]->height);
					image_erase(wx);
					image_erase(wy);
				}

				// energy minimization
				cout << "Backward flow estimation ..." << endl;
				Variational_MT minimzer_b;
				if(adaptCfg.exists("method") && adaptCfg.parameter("method").compare("forward") == 0)
					minimzer_b.one_direction = true;

				time(&t_start);
				minimzer_b.variational(wx, wy, im_back, thread_params);
				time(&t_stop);

				// write output file
				image_mul_scalar(wx, steps);		// scale flow
				image_mul_scalar(wy, steps);		// scale flow

				writeFlowFile(backward_flow_file, wx, wy);

				int time = (int) difftime(t_stop,t_start) + (int) difftime(ep_stop,ep_start) + t_preprocessing;
				// store results
				#pragma omp critical (sum)
				{
					avg_time += time;
					counter++;
				}

				// clean up
				image_delete(wx);
				image_delete(wy);

				cout << "Backward flow from frame " << start + f * skip << " to " << start + f * skip + steps * skip << " finished! (Computation took " << time << " s)" << endl;
			}  else
				cout << "Backward flow from frame " << start + f * skip << " to " << start + f * skip + steps * skip << " already exist!" << endl;
		}

		// clean up
		for(uint32_t f = start_f; f < end_f; f++) {
			color_image_delete(seq[f]);
			color_image_delete(un_seq[f]);
			delete[] img_files[f];
			// seq_back[f] only pointer!
		}
		color_image_delete(channel_weights);
		delete[] img_files;
		delete[] un_seq;
		delete[] un_seq_back;
		delete[] seq;
		delete[] seq_back;
		delete[] gt;
	}

	cout << "Done!" << endl;
    return 0;
}
