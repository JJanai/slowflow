/*
 * adaptiveFR.cpp
 *
 *  Created on: Sep 15, 2016
 *      Author: jjanai
 */
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "epic_flow_extended/image.h"
#include "epic_flow_extended/io.h"
#include "epic_flow_extended/epic.h"
#include "epic_flow_extended/variational.h"
#include "utils/utils.h"
#include "utils/parameter_list.h"
#include "configuration.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

enum COMPARISON {GROUNDTRUTH = 0, WARPING = 1};

/* show usage information */
void usage(){
    printf("usage:\n");
    printf("    ./adaptiveFR -path [path] -folder [specific folder | file with list] -format [file format] -scale [default 0.25] -skip [target frame (2)] -sample [number of estimation (10)] -step [frames between estimation (10)] -start [first frame (0)] -quantil -raw -overwrite -sintel -subframes -threads\n");
    printf("\n");
}

void setDefault(ParameterList& params) {
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

inline bool insideImg(double x, double y, int width, int height) {
	return (y >= 0 && y < height && x >= 0 && x < width);
}

int main(int argc, char **argv){
    if( argc < 2){
        if(argc>1) fprintf(stderr,"Error, not enough arguments\n");
        usage();
        exit(1);
    }

    // read optional arguments
    string format = "%07i.tif";
    uint32_t start = 0;

	#define isarg(key)  !strcmp(a,key)
	string path = "";
	string folder = "";
	string append = "";
	bool overwrite = false;

	int samples = 40;				// number of optical flow estimates
	int sample_step = 10;			// step size
	uint32_t all_frames = 2;		// frames for optical flow estimation (classical two frame formulation)
	int skip = 2;

	float q = 0.90f;				// quantil

	double scale = 0.25;
	double dm_scale = 1.0f;

	bool sintel = false;			// specific file names (we would like to be able to distinguish frame number from 24 fps and 1008 fps)
	bool subframes = false;			// are subframes specified

	bool raw = false;

	int threads = 1;

	int current_arg = 0;
	while(current_arg < argc ){
		const char* a = argv[current_arg++];
		if(a[0] != '-') {
			continue;
		}


		if( isarg("-h") || isarg("-help") )
			usage();
		else if( isarg("-path") )
			path = string(argv[current_arg++]);
		else if( isarg("-folder") )
			folder = string(argv[current_arg++]);
		else if( isarg("-threads") )
			threads = atoi(argv[current_arg++]);
		else if( isarg("-append") )
			append = string(argv[current_arg++]);
		else if( isarg("-scale") )
			scale = atof(argv[current_arg++]);
		else if( isarg("-skip") )
			skip = max(1, atoi(argv[current_arg++]));
		else if( isarg("-samples") )
			samples = atoi(argv[current_arg++]);
		else if( isarg("-step") )
			sample_step = atoi(argv[current_arg++]);
		else if( isarg("-start") )
			start = atoi(argv[current_arg++]);
		else if( isarg("-quantil") )
			q = atof(argv[current_arg++]);
		else if( isarg("-overwrite") )
			overwrite = true;
		else if( isarg("-sintel") )
			sintel = true;
		else if( isarg("-raw") )
			raw = true;
		else if( isarg("-subframes") )
			subframes = true;
		else if( isarg("-format") )
			format = string(argv[current_arg++]);
		else {
			fprintf(stderr, "unknown argument %s", a);
			usage();
			exit(1);
		}
	}

	vector<string> folders;
	if(folder.empty()) {
		fs::path apk_path(path + "/");
		boost::filesystem::directory_iterator dir(path + "/"), it, end;

		for(it = dir; it != end; it++)
		{
			const boost::filesystem::path& p = *it;
			folder = p.filename().string();
			if(boost::filesystem::is_directory(p) &&
				folder.compare("$RECYCLE.BIN") != 0 && folder.compare("preview") != 0 && folder.compare("Rallye") != 0 && folder.compare("System Volume Information") != 0 &&
				folder.compare("WDApps") != 0 && folder.c_str()[0] != '.') {

				folders.push_back(p.filename().string());
			}

		}
	} else {
		if(boost::filesystem::is_directory(path + "/" + folder + "/")) {
			folders.push_back(folder);
		} else {
			ifstream folder_input;
			string ifilename(folder);
			if (std::strcmp(ifilename.c_str(), "-") != 0) {
				folder_input.open(ifilename.c_str());

				if (!folder_input.is_open()) {
					std::cerr << ifilename << ": " << "no such file or directory" << "\n";
					return EXIT_FAILURE;
				}

				string line;

				// parse header
				while(getline(folder_input, line)) {
					if(boost::filesystem::is_directory(path + "/" + line + "/")) {
						folders.push_back(line);
					} else
						std::cerr << path + "/" + line + "/" << ": " << "no such directory" << "\n";
				}
			}
			folder_input.close();
		}
	}

	sort(folders.begin(), folders.end());

	if(sintel && !subframes)
		start = start * 1000;

	stringstream overview;
	#pragma omp parallel for num_threads(threads) schedule(static,1)
    for(uint32_t fidx = 0; fidx < folders.size(); fidx++) {
    	string thread_folder = folders[fidx];

		ParameterList params;
		setDefault(params);	// set default parameters

		// add input path and output path
		string sequence_path = "", output = "";
		params.file =  path + "/" + thread_folder + "/" + format;

		vector<int> seq_compression_params;
		seq_compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		seq_compression_params.push_back(0);
		seq_compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		seq_compression_params.push_back(100);

		params.Jets = 1;

		int start_format = (params.file.find_last_of('/') + 1);
		int end_format = params.file.length() - start_format;

		sequence_path = params.file.substr(0,start_format);
		string format = params.file.substr(start_format, end_format);
		if(sequence_path[sequence_path.length() - 1] != '/') sequence_path = sequence_path + "/";

		params.file = sequence_path;
		params.insert("format", format, true);

		// set output folder!
		output = sequence_path + "/adaptiveFR/";

		if(sequence_path.empty() || output.empty())
			continue;

		int len_format = format.find_last_of('.');
		string format_flow = format.substr(0,len_format);

		params.sequence_start = start;
		for(uint32_t i = 0; i < params.sequence_start_list.size(); i++)
			params.sequence_start_list[i] = start;


		if(output[output.length() - 1] != '/') output = output + "/";

		// set standard stinel params for epic flow
		epic_params_t epic_params;
		epic_params_default(&epic_params);
		variational_params_t flow_params;
		variational_params_default(&flow_params);
		epic_params.pref_nn= 25;
		epic_params.nn= 160;
		epic_params.coef_kernel = 1.1f;
		flow_params.niter_outer = 5;
		flow_params.alpha = 1.0f;
		flow_params.gamma = 0.72f;
		flow_params.delta = 0.0f;
		flow_params.sigma = 1.1f;

		// create results folder
		boost::filesystem::create_directories(output);
		boost::filesystem::create_directories(output+"tmp/");                // ParameterList result folder
		boost::filesystem::create_directories(output+"sequence/");                // ParameterList result folder


		image_t **wx = new image_t*[samples],
				**wy = new image_t*[samples];
		// TODO: TAKE SAMPLES FROM DIFFERENT STEPS
		for(int it = 0; it < samples; it++) {
			if(it > 0) {
				params.sequence_start += params.Jets * sample_step;
				for(uint32_t i = 0; i < params.sequence_start_list.size(); i++) {
					params.sequence_start_list[i] = params.sequence_start; 	//
				}
			}

			wx[it] = NULL;
			wy[it] = NULL;
			/*
			 * ################### read in image sequence ###################
			 */
			vector<int> red_loc = params.splitParameter<int>("raw_red_loc","1,0");

			char** img_files = new char*[all_frames];
			color_image_t **seq = new color_image_t*[all_frames];

			bool success = true;

			for(uint32_t f = 0; f < all_frames; f++) {
				char img_file[200];

				if(!sintel) {
					sprintf(img_file, (sequence_path+format).c_str(), params.sequence_start + f * skip);
				} else {
					int sintel_frame = params.sequence_start / 1000;
					int hfr_frame = f * skip + (params.sequence_start % 1000);

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

				if(access(img_file, F_OK) == -1) {
					cerr << "Could not find " << img_file << "!" << endl;
					success = false;
					break;
				}

				cout << "Reading " << img_file << "..." << endl;

				Mat img = imread(string(img_file), CV_LOAD_IMAGE_UNCHANGED);         // load images

				float norm = 1;
				if(img.type() == 2 || params.parameter<bool>("16bit", "0")) {
					norm = 1.0f/255;		// for 16 bit images
					params.insert("16bit", "1", true);
				}

				// convert to floating point
				img.convertTo(img, CV_32FC(img.channels()));

				/*
				 * DEMOSAICING
				 */
				if(raw) {
					Mat tmp = img.clone();
					color_image_t* tmp_in = color_image_new(img.cols, img.rows);
					color_image_t* tmp_out = color_image_new(img.cols, img.rows);

					switch(params.parameter<int>("raw_demosaicing", "0")) {
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

				// use only a part of the images
				if(params.extent.x > 0 || params.extent.y > 0) {
					img = img.rowRange(Range(params.center.y - params.extent.y/2,params.center.y + params.extent.y/2));
					img = img.colRange(Range(params.center.x - params.extent.x/2,params.center.x + params.extent.x/2));
				}

				// rescale image with gaussian blur to avoid anti-aliasing
				if(scale != 1) {
					GaussianBlur(img, img,Size(),1/sqrt(2*scale),1/sqrt(2*scale),BORDER_REPLICATE);
					resize(img, img, Size(0,0), scale, scale, INTER_LINEAR);
				}

				// print to file
				img_files[f] = new char[500];
				sprintf(img_files[f], (output+"sequence/frame_%i.png").c_str(), params.sequence_start + f * skip);
				Mat output_img;


				if(params.verbosity(WRITE_FILES)) {
					if(params.parameter<bool>("16bit", "0")) {
						img.convertTo(output_img, CV_16UC(img.channels()));
					} else {
						img.convertTo(output_img, CV_8UC(img.channels()), norm);
					}
					cv::cvtColor(output_img, output_img, CV_RGB2BGR);	// OpenCV uses BGR
					imwrite(img_files[f], output_img, seq_compression_params);
				}

				// use 8 bit for further processing
				img.convertTo(img, CV_8UC(img.channels()), norm);

				int width = img.cols;
				int height = img.rows;

				// copy data
				seq[f] = color_image_new(width, height);

				if(img.channels() == 1) {
					mat2colorImg<uchar>(img, seq[f]);
				} else
					colorMat2colorImg<Vec3b>(img, seq[f]);

				// resize and copy data for deep match
				GaussianBlur(img, img,Size(),1/sqrt(2*dm_scale),1/sqrt(2*dm_scale),BORDER_REPLICATE);
				resize(img, img, Size(0,0), dm_scale, dm_scale, INTER_LINEAR);

				sprintf(img_files[f], (output+"sequence/frame_epic_%i.png").c_str(), params.sequence_start + f * skip);
				output_img = Mat(img.rows, img.cols, CV_8UC(img.channels()));
				cv::cvtColor(img, output_img, CV_RGB2BGR);	// OpenCV uses BGR
				imwrite(img_files[f], output_img, seq_compression_params);
			}

			if(!success)
				continue;

			/*
			 *  write infos to file
			 */
			params.print();

			ofstream infos;
			infos.open((output + "config.cfg").c_str());
			infos << "# Epic Flow estimation\n";
			infos << params;
			infos.close();

			// write stats
			stringstream results;
			results << "frame\ttime\n\n";
			int avg_time = 0;
			int counter = 0;

			for(uint32_t j = 0; j < params.Jets; j++) {
				ParameterList thread_params(params);

				int f = j;

				color_image_t **im = &seq[f];

				// prepare variables
				wx[it] = image_new(im[0]->width*dm_scale, im[0]->height*dm_scale);
				wy[it] = image_new(im[0]->width*dm_scale, im[0]->height*dm_scale);

				time_t pp_start, pp_stop;
				int t_preprocessing = 0;
				char edges_f[1000], edges_cmd[1000], match_f[1000], match_cmd[1000], epic_cmd[1000], epic_f[1000];

				char forward_flow_file[200];
				if(!sintel)
					sprintf(forward_flow_file, (output + format_flow + ".flo").c_str(), params.sequence_start + f * skip);
				else
					sprintf(forward_flow_file, (output + format_flow + ".flo").c_str(), params.sequence_start + f * skip, 0);

				time_t t_start, t_stop;
				// skip finished frames
				if(overwrite || access( forward_flow_file, F_OK ) == -1) {
					/*
					 * ################### extract edges and get matches ###################
					 */
					cout << "Computing edges ..." << endl;
					sprintf(edges_f, (output+"tmp/edges_%i.dat").c_str(), params.sequence_start + f);

					if(overwrite || access( edges_f, F_OK ) == -1) {
						sprintf(edges_cmd, "matlab -nodesktop -nojvm -r \"addpath(\'%s/matlab/\'); detect_edges(\'%s\',\'%s\'); exit\"", SOURCE_PATH.c_str(), img_files[j], edges_f);

						time(&pp_start);
						system(edges_cmd);
						// Call the function
						time(&pp_stop);
						t_preprocessing += (int) difftime(pp_stop, pp_start);
					}

					// matches to target frame
					cout << "Computing matches between " << params.sequence_start + f * skip << " and " << params.sequence_start + (f + 1) * skip<< " ..." << endl;
					sprintf(match_f, (output+"tmp/matches_%i_%i.dat").c_str(), params.sequence_start + f * skip, params.sequence_start + (f + 1) * skip);

					cout << img_files[j] << " and " << img_files[j + 1] << endl;
					if(overwrite || access( match_f, F_OK ) == -1) {
						sprintf(match_cmd, "%s/deepmatching %s %s -png_settings -out %s", DEEPMATCHING_PATH.c_str(), img_files[j], img_files[j + 1], match_f);

						time(&pp_start);
						system(match_cmd);
						time(&pp_stop);
						t_preprocessing += (int) difftime(pp_stop, pp_start);
					}


					/*
					 * ############ forward flow ##################
					 */
					cout << "Forward flow estimation ..." << endl;

					// matches to target frame
					float_image forward_edges = read_edges(edges_f, im[0]->width, im[0]->height);
					time(&pp_start);
					float_image forward_matches = read_matches(match_f);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);

					color_image_t *imlab = rgb_to_lab(im[0]);

					// initilize with deep matches
					cout << "Epic interpolation of forward flow ..." << endl;
					time(&pp_start);
					epic(wx[it], wy[it], imlab, &forward_matches, &forward_edges, &epic_params, 1);
					time(&pp_stop);
					t_preprocessing += (int) difftime(pp_stop, pp_start);

					// energy minimization
					time(&t_start);
					variational(wx[it], wy[it], im[0], im[1], &flow_params);
					system(epic_cmd);
					time(&t_stop);
					t_preprocessing += difftime(t_stop, t_start);

					color_image_delete(imlab);

					free(forward_matches.pixels);
					free(forward_edges.pixels);

				    // write output file
				    writeFlowFile(forward_flow_file, wx[it], wy[it]);

					cout << "Forward flow from frame " << params.sequence_start + f * skip << " to " << params.sequence_start + (f + 1) * skip << " finished! (Computation took " << t_preprocessing<< " s)" << endl;
				} else {
					image_t **tmp;
					tmp = readFlowFile(forward_flow_file);

					wx[it] = tmp[0];
					wy[it] = tmp[1];

					cout << "Forward flow from frame " << params.sequence_start + f * skip << " to " << params.sequence_start + (f + 1) * skip << " already exist!" << endl;
				}

				Mat floImg = flowColorImg(wx[it], wy[it], params.verbosity(VER_CMD));
				// write flow image to file
				if(!floImg.data) {                 // Check for invalid input
					cout <<  "No forward flow for frame " << params.sequence_start + f * skip << std::endl ;
				} else {
					if(!output.empty()) {
						stringstream flowF;
						flowF << output <<  "tmp/frame_" << params.sequence_start + f * skip << ".png";

						imwrite((flowF.str()), floImg);
					}
				}

				// normalize flow to recorded resolution and frame rate
				image_mul_scalar(wx[it], 1.0f / (scale * skip));		// scale flow
				image_mul_scalar(wy[it], 1.0f / (scale * skip));		// scale flow
			}


			if(counter > 0) avg_time /= counter;

			cout << "Average computation was " << avg_time << " s" << endl;
			results << "\n\navg\t" << avg_time << "s\n";

			// write experiment results to file
			if(!params.output.empty() && counter > 0) {
				ofstream infos;
				infos.open((output + "results.info").c_str());
				infos << "Epic Flow Multi Frame\n";
				infos << "\n";
				infos << results.str();
				infos.close();
			}

			// clean up
			for(uint32_t f = 0; f < all_frames; f++) {
				color_image_delete(seq[f]);
				delete[] img_files[f];
			}
			delete[] img_files;
			delete[] seq;
		}

		/*
		 * ########################################### compute quantil ##############################################
		 */
		int used = 0;
		vector<double> magnitudes;
		magnitudes.reserve(samples * wx[0]->height * wx[0]->width);
		for(int it = 0; it < samples; it++) {
			if(wx[it] == NULL || wy[it] == NULL) continue;

			for (int y = 0; y < wx[it]->height; y++) {
				for (int x = 0; x < wx[it]->width; x++) {
					magnitudes.push_back(sqrt(wx[it]->data[y * wx[it]->stride + x] * wx[it]->data[y * wx[it]->stride + x] + wy[it]->data[y * wy[it]->stride + x] * wy[it]->data[y * wy[it]->stride + x]));
				}
			}

			used++;
		}
		sort(magnitudes.begin(), magnitudes.end());

		float np = q * magnitudes.size() - 1;
		double quantil = 0;
		if((np < magnitudes.size() - 1) && fmod(np,2.0f) == 0) {
			quantil = 0.5f * (magnitudes[(int) np] + magnitudes[(int) np + 1]);
		} else {
			quantil = (magnitudes[(int) ceil(np)]);
		}

		double maxq = magnitudes.back();

		cout << "Quantil: " << quantil << endl;

		// write experiment results to file
		ofstream infos;
		infos.open((output + "results.info").c_str());
		infos << "Adaptive Frame rate\n";
		infos << "\n";
		infos << "samples	" << used << "\n";
		infos << "sample_step	" << sample_step << "\n";
		infos << "skip	" << skip << "\n";
		infos << q << " quantil	" << quantil << "\n";
		infos << "max	" << maxq << "\n";
		infos.close();

		#pragma omp critical (overview)
		{
			overview << thread_folder << "\t" << q << " quantil\t" << quantil << "\n";
		}

		string fname = sequence_path + "quantil.dat";
		if(!append.empty())
			infos.open(append.c_str(), std::ofstream::out | std::ofstream::app);
		else
			infos.open(fname.c_str());
		infos << quantil << "\n";
		infos << maxq << "\n";
		infos.close();

		for(int it = 0; it < samples; it++) {
			image_delete(wx[it]);
			image_delete(wy[it]);
		}
		delete[] wx;
		delete[] wy;
    }

	ofstream infos;
	infos.open((path + "results.info").c_str());
	infos << "Adaptive Frame rate\n";
	infos << "\n";
	infos << "samples	" << samples << "\n";
	infos << "sample_step	" << sample_step << "\n";
	infos << "skip	" << skip << "\n\n";
	infos << overview.str();
	infos.close();

	cout << "Done!" << endl;
    return 0;
}
