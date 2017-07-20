/*
 * utils.cpp
 *
 *  Created on: Mar 22, 2016
 *      Author: jjanai
 */
#include "utils.h"

void dx(const float* img, float* der, int width, int height, int stride) {
    // iterate over each pixel
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            // dI/dx = I(x + 1) - I(x - 1)
        	float c1_p1 = img[y*stride + x];
        	float c1_m1 = img[y*stride + x];
        	if(x+1 < width) c1_p1 = img[y*stride + x + 1];
        	if(x-1 > 0) c1_m1 = img[y*stride + x - 1];
            // symmetric gradient
        	der[y*stride+x] = 0.5 * (c1_p1 - c1_m1);
        }
    }
}

void dy(const float* img, float* der, int width, int height, int stride) {
    // iterate over each pixel
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            // dI/dx = I(x + 1) - I(x - 1)
        	float c1_p1 = img[y*stride + x];
        	float c1_m1 = img[y*stride + x];
        	if(y+1 < height) c1_p1 = img[(y + 1)*stride + x];
        	if(y-1 > 0) c1_m1 = img[(y - 1)*stride + x];
            // symmetric gradient
        	der[y*stride+x] = 0.5 * (c1_p1 - c1_m1);
        }
    }
}

double computeEPE(const image_t* flow_x, const image_t* flow_y, const image_t* gt_x, const image_t* gt_y, Mat* error_img, Mat* mask, double norm) {
	if(flow_x->height != gt_x->height || flow_x->width != gt_x->width) {
		cerr << "Error (computetEPE): Dimension do not fit between ground truth and estimation!" << endl;
		return -1;
	}

	if(error_img != NULL) error_img->convertTo(*error_img, CV_32FC1);

	double epe = 0;
	int counter = 0;
	double maxi = 0;
	double maxerr = 0;
	for(int y = 0; y < gt_x->height; y++) {
		for(int x = 0; x < gt_x->width; x++) {
			// skip pixel in case of
			if (mask != NULL && mask->at<double>(y,x) == 0) continue;
            if (fabs(gt_x->data[y * gt_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(gt_y->data[y * gt_y->stride + x]) > UNKNOWN_FLOW_THRESH) continue;
            if (fabs(flow_x->data[y * flow_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(flow_y->data[y * flow_y->stride + x]) > UNKNOWN_FLOW_THRESH) continue;

			float t1 = flow_x->data[y * gt_x->stride + x] - gt_x->data[y * gt_x->stride + x];
			float t2 = flow_y->data[y * gt_x->stride + x] - gt_y->data[y * gt_x->stride + x];

			float err = sqrt(t1*t1 + t2*t2);

			epe += err;
			counter++;

			if(error_img != NULL) error_img->at<float>(y,x) = err;

			maxi = std::max(maxi, (double) sqrt(gt_x->data[y * gt_x->stride + x] * gt_x->data[y * gt_x->stride + x] + gt_y->data[y * gt_y->stride + x] * gt_y->data[y * gt_y->stride + x]));
			maxerr = std::max(maxerr, (double) err);
		}
	}

	if(norm > 0)
		maxi = norm;

	if(error_img != NULL)  {
		// draw legend
		int start_y = 10, end_y = start_y + 10, start_x = gt_x->width - 120, end_x = start_x + 100;
		for (int y = start_y; y < end_y; y++) {
			for (int x = start_x; x < end_x; x++) {
				error_img->at<float>(y,x) = ((x - start_x) / (end_x - start_x - 1.0f)) * maxerr;
			}
		}

		// write text for legend
		stringstream text;
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.3;
		int thickness = 1;
		int baseline = 0;

		text << 0;
		Size textSize = getTextSize(text.str(), fontFace, fontScale, thickness, &baseline);
		Point textOrg(start_x, end_y + 15);
		putText(*error_img, text.str(), textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		text.str("");
		text << round(maxerr*100) / 100.0f;
		textSize = getTextSize(text.str(), fontFace, fontScale, thickness, &baseline);
		textOrg = Point(end_x - textSize.width, end_y + 15);
		putText(*error_img, text.str(), textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		error_img->convertTo(*error_img, CV_8UC1, 255/maxi);
	}

	if(counter != 0) epe = epe / counter;

	return epe;
}

double computeAAE(const image_t* flow_x, const image_t* flow_y, const image_t* gt_x, const image_t* gt_y, Mat* mask) {
	if(flow_x->height != gt_x->height || flow_x->width != gt_x->width) {
		cerr << "Error (computetEPE): Dimension do not fit between ground truth and estimation!" << endl;
		return -1;
	}

	double aae = 0;
	int counter = 0;
	for(int y = 0; y < gt_x->height; y++) {
		for(int x = 0; x < gt_x->width; x++) {
			// skip pixel in case of
			if (mask != NULL && mask->at<double>(y,x) == 0) continue;
            if (fabs(gt_x->data[y * gt_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(gt_y->data[y * gt_y->stride + x]) > UNKNOWN_FLOW_THRESH) continue;
            if (fabs(flow_x->data[y * flow_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(flow_y->data[y * flow_y->stride + x]) > UNKNOWN_FLOW_THRESH) continue;

			double n1 = sqrt(flow_x->data[y * flow_x->stride + x]*flow_x->data[y * flow_x->stride + x] + flow_y->data[y * flow_y->stride + x]*flow_y->data[y * flow_y->stride + x] + 1.0f*1.0f);
			double n2 = sqrt(gt_x->data[y * gt_x->stride + x]*gt_x->data[y * gt_x->stride + x] + gt_y->data[y * gt_y->stride + x]*gt_y->data[y * gt_y->stride + x] + 1.0f*1.0f);

			double t1 = (flow_x->data[y * flow_x->stride + x] * gt_x->data[y * gt_x->stride + x]);
			double t2 = (flow_y->data[y * flow_y->stride + x] * gt_y->data[y * gt_y->stride + x]);
			double t3 = 1.0;

			aae += acos(min((t1 + t2 + t3) / (n1 * n2), 1.0));

			counter++;
		}
	}

	if(counter != 0) aae = aae / counter;

	return aae;
}

double computeRMS(const color_image_t* im1, const color_image_t* im2, const image_t* flow_x, const image_t* flow_y) {
	if(im1->height != im2->height || im1->width != im2->width) {
		cerr << "Error (computetEPE): Dimension do not fit between ground truth and estimation!" << endl;
		return -1;
	}

	double rms = 0;
	int counter = 0;
	for(int y = 0; y < im1->height; y++) {
		for(int x = 0; x < im1->width; x++) {
            if (fabs(flow_x->data[y * flow_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(flow_y->data[y * flow_y->stride + x]) > UNKNOWN_FLOW_THRESH) continue;

			float t1 = im1->c1[y * im1->stride + x] - im2->c1[y * im2->stride + x];
			float t2 = im1->c2[y * im1->stride + x] - im2->c2[y * im2->stride + x];
			float t3 = im1->c3[y * im1->stride + x] - im2->c3[y * im2->stride + x];

			rms += sqrt(t1*t1 + t2*t2 + t3*t3);

			counter++;
		}
	}

	return rms / (counter);
}

Mat removeSmallSegments(Mat& F, float similarity_threshold, int min_segment_size) {
  // get image width and height
  int width = F.cols;
  int height = F.rows;
  Mat V(height, width, CV_8SC1, Scalar::all(255));	// everything valid

  // allocate memory on heap for dynamic programming arrays
  int32_t *I_done     = (int32_t*)calloc(width*height,sizeof(int32_t));
  int32_t *seg_list_u = (int32_t*)calloc(width*height,sizeof(int32_t));
  int32_t *seg_list_v = (int32_t*)calloc(width*height,sizeof(int32_t));
  int32_t seg_list_count;
  int32_t seg_list_curr;
  int32_t u_neighbor[4];
  int32_t v_neighbor[4];
  int32_t u_seg_curr;
  int32_t v_seg_curr;

  // declare loop variables
  int32_t addr_start, addr_curr, addr_neighbor;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {

      // get address of first pixel in this segment
      addr_start = getAddressOffsetImage(u,v,width);

      // if this pixel has not already been processed
      if (*(I_done+addr_start)==0) {

        // init segment list (add first element
        // and set it to be the next element to check)
        *(seg_list_u+0) = u;
        *(seg_list_v+0) = v;
        seg_list_count  = 1;
        seg_list_curr   = 0;

        // add neighboring segments as long as there
        // are none-processed pixels in the seg_list;
        // none-processed means: seg_list_curr<seg_list_count
        while (seg_list_curr<seg_list_count) {

          // get current position from seg_list
          u_seg_curr = *(seg_list_u+seg_list_curr);
          v_seg_curr = *(seg_list_v+seg_list_curr);

          // get address of current pixel in this segment
          addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,width);

          // fill list with neighbor positions
          u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
          u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
          u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
          u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;

          // for all neighbors do
          for (int32_t i=0; i<4; i++) {

            // check if neighbor is inside image
            if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<width && v_neighbor[i]<height) {

              // get neighbor pixel address
              addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],width);

              // check if neighbor has not been added yet and if it is valid
              if (*(I_done+addr_neighbor)==0) {

                // is the neighbor similar to the current pixel
                // (=belonging to the current segment)
                  if (abs(F.at<int>(v_seg_curr, u_seg_curr) - F.at<int>(v_neighbor[i], u_neighbor[i]))
                      <= similarity_threshold) {

                  // add neighbor coordinates to segment list
                  *(seg_list_u+seg_list_count) = u_neighbor[i];
                  *(seg_list_v+seg_list_count) = v_neighbor[i];
                  seg_list_count++;

                  // set neighbor pixel in I_done to "done"
                  // (otherwise a pixel may be added 2 times to the list, as
                  //  neighbor of one pixel and as neighbor of another pixel)
                  *(I_done+addr_neighbor) = 1;
                }
              }

            }
          }

          // set current pixel in seg_list to "done"
          seg_list_curr++;

          // set current pixel in I_done to "done"
          *(I_done+addr_curr) = 1;

        } // end: while (seg_list_curr<seg_list_count)

        // if segment NOT large enough => invalidate pixels
        if (seg_list_count<min_segment_size) {

          // for all pixels in current segment invalidate pixels
          for (int32_t i=0; i<seg_list_count; i++) {
            V.at<uchar>(seg_list_v[i],seg_list_u[i]) = 0;
            F.at<int>(seg_list_v[i],seg_list_u[i]) = 0;
          }
        }
      }

    }
  }

  // free memory
  free(I_done);
  free(seg_list_u);
  free(seg_list_v);

  return V;
}

void warp_image(const color_image_t* img, const image_t* flow_x, const image_t* flow_y, color_image_t* warped, float scale) {
	for(int y = 0; y < img->height; y++) {
		for(int x = 0; x < img->width; x++) {
            if (fabs(flow_x->data[y * flow_x->stride + x]) > UNKNOWN_FLOW_THRESH || fabs(flow_y->data[y * flow_y->stride + x]) > UNKNOWN_FLOW_THRESH) {
				warped->c1[y * flow_x->stride  + x] = 0;
				warped->c2[y * flow_x->stride  + x] = 0;
				warped->c3[y * flow_x->stride  + x] = 0;
            	continue;
            }

			double w_x = x - scale * flow_x->data[y * flow_x->stride  + x]; // for pixel x,y in target frame -(u,v) gives the reference frame
			double w_y = y - scale * flow_y->data[y * flow_y->stride  + x];

			if(w_x >= 0 && w_x < img->width && w_y >= 0 && w_y < img->height) {
				warped->c1[y * flow_x->stride  + x] = bilinearInterp(w_x, w_y, img->c1, img->height, img->width, img->stride);
				warped->c2[y * flow_x->stride  + x] = bilinearInterp(w_x, w_y, img->c2, img->height, img->width, img->stride);
				warped->c3[y * flow_x->stride  + x] = bilinearInterp(w_x, w_y, img->c3, img->height, img->width, img->stride);
			}
		}
	}
}

Mat crop(Mat src, Point center, Point extent) {
	Mat subset = Mat::zeros(extent.y, extent.x, CV_64FC2);
	for (int y =  0; y < extent.y; y++) {
		for (int x =  0; x < extent.x; x++) {
			int off_y = y - extent.y/2 + center.y ;
			int off_x = x - extent.x/2 + center.x ;
			subset.at<Vec2d>(y,x) = src.at<Vec2d>(off_y,off_x);
		}
	}
	return subset;
}

void MatToCFImg(const Mat& src, CFloatImage& output) {
    Mat tmp = src.clone();

    int height = src.rows;
    int width = src.cols;
    int nBands = src.channels();

    if(output.Shape().height != height || output.Shape().width != width) {
    	cerr << "utils.cpp:MatToCFImg: Dimesion of Mat and CFloatImage do not match!" << endl;
    	return;
    }

    // change the order of channels to first channel flow u and second channel flow v
    tmp.convertTo(tmp, CV_32FC(nBands));       // convert to float
    vector<Mat> chans;
    split(tmp, chans);
    reverse(chans.begin(), chans.end());
    merge(chans, tmp);

    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        float* ptrMat = (float*) tmp.row(y).data;
        float* ptrCFImg = &output.Pixel(0, y, 0);

        memcpy(ptrCFImg, ptrMat, sizeof(float) * n);
    }
}

void CFImgToMat(CFloatImage &src, Mat& output) {
    CShape sh = src.Shape();
    int width = sh.width;
    int height = sh.height;
    int nBands = sh.nBands;

    output = Mat(height, width, CV_32FC(nBands));

    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        float* ptrMat = (float*) output.row(y).data;
        float* ptrCFImg = &src.Pixel(0, y, 0);

        memcpy(ptrMat, ptrCFImg, sizeof(float) * n);
    }

    // change the order of channels to first channel flow v and second channel flow u
    output.convertTo(output, CV_64FC2);       // convert to double
    vector<Mat> chans;
    split(output, chans);
    reverse(chans.begin(), chans.end());
    merge(chans, output);
}

void cvArrow(Mat Image, int x, int y, int u, int v, CvScalar Color, int Size, int Thickness) {
	cv::Point pt1, pt2;
	double Theta;

	if (u == 0)
		Theta = PI / 2;
	else
		Theta = atan2(double(v), (double) (u));

	pt1.x = x;
	pt1.y = y;

	pt2.x = x + u;
	pt2.y = y + v;

	Size = (int) (Size * 0.707);

	if (Theta == PI / 2 && pt1.y > pt2.y) {
		pt2.x = x;
		pt2.y = y;

		pt1.x = (int) (Size * cos(Theta) - Size * sin(Theta) + pt2.x);
		pt1.y = (int) (Size * sin(Theta) + Size * cos(Theta) + pt2.y);
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line

		pt1.x = (int) (Size * cos(Theta) + Size * sin(Theta) + pt2.x);
		pt1.y = (int) (Size * sin(Theta) - Size * cos(Theta) + pt2.y);
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line
	} else {
		pt2.x = x;
		pt2.y = y;

		pt1.x = (int) (-Size * cos(Theta) - Size * sin(Theta) + pt2.x);
		pt1.y = (int) (-Size * sin(Theta) + Size * cos(Theta) + pt2.y);
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line

		pt1.x = (int) (-Size * cos(Theta) + Size * sin(Theta) + pt2.x);
		pt1.y = (int) (-Size * sin(Theta) - Size * cos(Theta) + pt2.y);
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line
	}

}

double bilinearInterp(double x, double y, const float* fct, int height, int width, int stride) {
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

    double f_x0y0 = fct[stride * y0 + x0];
    double f_x1y0 = fct[stride * y0 + x1];
    double f_x0y1 = fct[stride * y1 + x0];
    double f_x1y1 = fct[stride * y1 + x1];

    return (1 - weight_y) * (1 - weight_x) * f_x0y0
           + (1 - weight_y) * weight_x * f_x1y0
           + weight_y * (1 - weight_x) * f_x0y1
           + weight_y * weight_x * f_x1y1;
}

void forwardBackwardCheck(Mat& mask, const Mat *forward, const Mat *backward, uint32_t FF, float epsilon, bool jetwise) {
	uint32_t height = forward[0].rows;
	uint32_t width = forward[0].cols;

	// create consistency mask with 1 == consistent
	if(mask.rows != (int) height && mask.cols != (int) width) mask = Mat::zeros(height, width, CV_8SC1);

	// accumulate forward and backward flow of all subsets
	if(!jetwise) {
		// accumulated forward and backward flow
		Mat acc_forward_flow = Mat::zeros(height, width, CV_64FC2);
		Mat acc_backward_flow = Mat::zeros(height, width, CV_64FC2);

		for(uint32_t f = 0; f < FF; f++) {
			for(u_int32_t y = 0; y < height; y++) {
				for(u_int32_t x = 0; x < width; x++) {
					Vec2d f_corr = acc_forward_flow.at<Vec2d>(y,x) + Vec2d(y,x);
					if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
						acc_forward_flow.at<Vec2d>(y,x) += Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward[f],0),				// bilinear interpolation
																 bilinearInterp<double>(f_corr[1],f_corr[0],forward[f],1));
					}

					Vec2d b_corr = acc_backward_flow.at<Vec2d>(y,x) + Vec2d(y,x);
					if(b_corr[0] >= 0 && b_corr[0] < height && b_corr[1] >= 0 && b_corr[1] < width) {
						acc_backward_flow.at<Vec2d>(y,x) += Vec2d(bilinearInterp<double>(b_corr[1],b_corr[0],backward[(FF-1) - f],0),
																  bilinearInterp<double>(b_corr[1],b_corr[0],backward[(FF-1) - f],1));	// bilinear interpolation
					}
				}
			}
		}

		// check forward backward consistency of accumulated flow
		for(u_int32_t y = 0; y < height; y++) {
			for(u_int32_t x = 0; x < width; x++) {

				Vec2d f_corr = acc_forward_flow.at<Vec2d>(y,x) + Vec2d(y,x);

				if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
					Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],acc_backward_flow,0),
									  bilinearInterp<double>(f_corr[1],f_corr[0],acc_backward_flow,1));	// bilinear interpolation

					Vec2d diff = acc_forward_flow.at<Vec2d>(y,x) + vec;

					if(sqrt(diff[0]*diff[0] + diff[1]*diff[1]) > epsilon) mask.at<uchar>(y,x) = 1;
				} else {
					mask.at<uchar>(y,x) = 1;
				}
			}
		}
	} else {

		for(uint32_t f = 0; f < FF; f++) {
			for(u_int32_t y = 0; y < height; y++) {
				for(u_int32_t x = 0; x < width; x++) {
					Vec2d f_corr = forward[f].at<Vec2d>(y,x) + Vec2d(y,x);

					Vec2d diff = forward[f].at<Vec2d>(y,x);
					if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width)
						diff += Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],backward[f],0),
									  bilinearInterp<double>(f_corr[1],f_corr[0],backward[f],1));	// bilinear interpolation

					if(sqrt(diff[0]*diff[0] + diff[1]*diff[1]) > epsilon) mask.at<uchar>(y,x) = 1;
				}
			}
		}
	}
}

Mat accumulateConsistentBatches(Mat *acc_forward, Mat *forward_flow, Mat *backward_flow, Mat* occlusions, uint32_t FF, double epsilon, uint32_t skip, bool discard, bool verbose) {
    int oheight = forward_flow[0].rows;
    int owidth = forward_flow[0].cols;

    int xy_incr = skip + 1;
    int xy_start = 0.5f * skip;
    uint32_t height = floor((1.0f * oheight) / xy_incr);
	uint32_t width = floor((1.0f * owidth) / xy_incr);

    // accumulate forward and backward flow of all subsets
	if(verbose)
		cout << "Accumulating flow ";

	Mat last_flow = Mat::zeros(height, width, CV_64FC2);						// avoid zero flow if directly occluded!
	for(u_int32_t y = 0; y < height; y++) {
		for(u_int32_t x = 0; x < width; x++) {
			last_flow.at<Vec2d>(y,x) = forward_flow[0].at<Vec2d>(y * xy_incr + xy_start, x * xy_incr + xy_start);
		}
	}

	Mat occluded = Mat::zeros(height, width, CV_8SC1);
	Mat tracked = Mat(height, width, CV_32SC1, Scalar::all(FF));		// set all to fulled tracked!!!!!!!!!!!!!!!!

	for(uint32_t f = 0; f < FF; f++) {
		acc_forward[f] = Mat::zeros(height, width, CV_64FC2);
		if(verbose)
			cout << "from " << f << " to " << f + 1 << ", ";

		for(u_int32_t y = 0; y < height; y++) {
			for(u_int32_t x = 0; x < width; x++) {
				if(occluded.at<uchar>(y,x) == 1)
					continue;

				Vec2d f_corr = Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);
				if(f > 0) {
					f_corr += acc_forward[f-1].at<Vec2d>(y,x);
					acc_forward[f].at<Vec2d>(y,x) = acc_forward[f-1].at<Vec2d>(y,x);
				}

				if(f_corr[0] >= 0 && f_corr[0] < oheight && f_corr[1] >= 0 && f_corr[1] < owidth) {
					if (occlusions != NULL && occlusions[f].at<uchar>(f_corr[0], f_corr[1]) == 0) {
						occluded.at<uchar>(y,x) = 1;

						// change only once
						if( tracked.at<int>(y,x) == (int) FF) {
							if(discard)
								tracked.at<int>(y,x) = 0;
							else
								tracked.at<int>(y,x) = f+1;
						}
					}


					Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward_flow[f],0),				// bilinear interpolation
									  bilinearInterp<double>(f_corr[1],f_corr[0],forward_flow[f],1));

					Vec2d f_corr_n = f_corr + vec;
					Vec2d diff = vec - last_flow.at<Vec2d>(y,x); // TODO: OUT OF IMAGE CONFIDENCE (FOR NOW CONSTANT VEL)
					if(f_corr_n[0] >= 0 && f_corr_n[0] < oheight && f_corr_n[1] >= 0 && f_corr_n[1] < owidth)
						diff = vec + Vec2d(bilinearInterp<double>(f_corr_n[1],f_corr_n[0],backward_flow[f],0), bilinearInterp<double>(f_corr_n[1],f_corr_n[0],backward_flow[f],1));	// bilinear interpolation


					double err = sqrt(diff[0]*diff[0] + diff[1]*diff[1]);

					if(err > epsilon) {
						//  Constant Velocity!
						acc_forward[f].at<Vec2d>(y,x) += last_flow.at<Vec2d>(y,x);

						// change only once
						if( tracked.at<int>(y,x) == (int) FF) {
							if(discard)
								tracked.at<int>(y,x) = 0;
							else
								tracked.at<int>(y,x) = f+1;
						}
					} else {
						acc_forward[f].at<Vec2d>(y,x) += vec;

						last_flow.at<Vec2d>(y,x) = vec;
					}
				} else {
					//  Constant Velocity!
					acc_forward[f].at<Vec2d>(y,x) += last_flow.at<Vec2d>(y,x);

					// change only once
					if(tracked.at<int>(y,x) == (int) FF) {
						if(discard)
							tracked.at<int>(y,x) = 0;
						else
							tracked.at<int>(y,x) = f+1;
					}
				}
			}
		}
	}

    if(verbose)
    	cout << endl;

    return tracked;
}

void forwardBackwardConsistency(Mat *forward_flow, Mat *backward_flow, Mat& mask, Mat& flow_diff, uint32_t FF, int S, double epsilon, bool jetwise, uint32_t skip, bool verbose, int threads) {
    int oheight = forward_flow[0].rows;
    int owidth = forward_flow[0].cols;

    int steps = (S-1);

    int xy_incr = skip + 1;
    int xy_start = 0.5f * skip;
    uint32_t height = floor((1.0f * oheight) / xy_incr);
    uint32_t width = floor((1.0f * owidth) / xy_incr);

    // create consistency mask with 1 == consistent
    if(mask.rows != (int) height && mask.cols != (int) width) mask = Mat::ones(height, width, CV_64FC1);
    flow_diff = Mat::zeros(height, width, CV_64FC1);

    // check if number of frames sufficient
    int mod = FF % steps;
    if(mod > 0) {
        cerr << "Forward backward consistency check with only " << FF - mod << " flow fields since it is not possible with " << FF << " flow fields!" << endl;
        FF -= mod;
    }

    // accumulate forward and backward flow of all subsets
    if(!jetwise) {
        // accumulated forward and backward flow
        Mat acc_forward_flow = Mat::zeros(height, width, CV_64FC2);
        Mat acc_backward_flow = Mat::zeros(height, width, CV_64FC2);

        if(verbose)
        	cout << "Accumulating flow ";

		for(uint32_t f = (steps-1); f < FF; f+=steps) {
	        if(verbose)
	        	cout << "from " << f - steps + 1 << " to " << f + 1 << ", ";

			#pragma omp parallel for num_threads(threads)
			for(u_int32_t y = 0; y < height; y++) {
				for(u_int32_t x = 0; x < width; x++) {
					Vec2d f_corr = acc_forward_flow.at<Vec2d>(y,x) + Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);
					if(f_corr[0] >= 0 && f_corr[0] < oheight && f_corr[1] >= 0 && f_corr[1] < owidth) {
						acc_forward_flow.at<Vec2d>(y,x) += Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward_flow[f],0),				// bilinear interpolation
																 bilinearInterp<double>(f_corr[1],f_corr[0],forward_flow[f],1));
					}

					Vec2d b_corr = acc_backward_flow.at<Vec2d>(y,x) + Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);
					if(b_corr[0] >= 0 && b_corr[0] < oheight && b_corr[1] >= 0 && b_corr[1] < owidth) {
						acc_backward_flow.at<Vec2d>(y,x) += Vec2d(bilinearInterp<double>(b_corr[1],b_corr[0],backward_flow[(FF-1) - f],0),
																  bilinearInterp<double>(b_corr[1],b_corr[0],backward_flow[(FF-1) - f],1));	// bilinear interpolation
					}
				}
			}
		}

		// check forward backward consistency of accumulated flow
		#pragma omp parallel for num_threads(threads)
		for(u_int32_t y = 0; y < height; y++) {
			for(u_int32_t x = 0; x < width; x++) {
				flow_diff.at<double>(y,x) = INT_MAX;

				Vec2d f_corr = acc_forward_flow.at<Vec2d>(y,x) + Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);
				f_corr /= xy_incr;

				if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
                	Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],acc_backward_flow,0),
                					  bilinearInterp<double>(f_corr[1],f_corr[0],acc_backward_flow,1));	// bilinear interpolation

					Vec2d diff = acc_forward_flow.at<Vec2d>(y,x) + vec;

					flow_diff.at<double>(y,x) = sqrt(diff[0]*diff[0] + diff[1]*diff[1]);

					if(flow_diff.at<double>(y,x) > epsilon) mask.at<double>(y,x) = 0;
				}
			}
		}
    } else {
        if(verbose)
        	cout << "Checking flow ";

		for(uint32_t f = 0; f < FF; f+=steps) {
	        if(verbose)
	        	cout << "from " << f << " to " << f + 1 << ", "  << flush;

			#pragma omp parallel for shared(forward_flow,backward_flow) num_threads(threads)
			for(u_int32_t y = 0; y < height; y++) {
				for(u_int32_t x = 0; x < width; x++) {
					Vec2d f_corr = steps * forward_flow[f].at<Vec2d>(y,x) + Vec2d(y,x);
					f_corr /= xy_incr;

					Vec2d diff = forward_flow[f].at<Vec2d>(y,x);
					if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width)
						diff += Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],backward_flow[f],0),
	                				  bilinearInterp<double>(f_corr[1],f_corr[0],backward_flow[f],1));	// bilinear interpolation

					flow_diff.at<double>(y,x) = sqrt(diff[0]*diff[0] + diff[1]*diff[1]);

					if(flow_diff.at<double>(y,x) > epsilon) mask.at<double>(y,x) = 0;
				}
			}
		}
    }

    if(verbose)
    	cout << endl;
}

Mat fuseOcclusions(const Mat* forward, const Mat *occlusions, int start, int length) {
	int height = forward[0].rows;
	int width = forward[0].cols;

	Mat acc_flow = Mat::zeros(height, width, CV_64FC2);
	Mat occluded = Mat::zeros(height, width, CV_8UC1);

	for(int i = 0; i < length; i++) {;
		// adding accumulated flow to flow of subsequence
		Mat incr = Mat::zeros(height, width, CV_64FC2);
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				// pixel lies in occlusion
				if(occluded.at<uchar>(y,x) == 1) {
					continue;
				}

				Vec2d f_corr = acc_flow.at<Vec2d>(y,x) + Vec2d(y,x);	// compute pixel in frame i

				if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
					int r_y = max(0, min(height - 1,(int) round(f_corr[0])));
					int r_x = max(0, min(width - 1,(int) round(f_corr[1])));

					// pixel becomes occluded
					if(occlusions[start + i].at<uchar>(r_y, r_x) != 0) {
						occluded.at<uchar>(y,x) = 1;
						continue;
					}

					Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward[start + i],0),
									  bilinearInterp<double>(f_corr[1],f_corr[0],forward[start + i],1));	// bilinear interpolation

					incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + vec;
				} else
					occluded.at<uchar>(y,x) = 1;
			}
		}

		acc_flow = incr;
	}

	return occluded;
}

Mat fuseOcclusions( image_t*** forward, const Mat *occlusions, int start, int length) {
	int height = forward[0][0]->height;
	int width = forward[0][0]->width;

	Mat acc_flow = Mat::zeros(height, width, CV_64FC2);
	Mat occluded = Mat::zeros(height, width, CV_8SC1);

	for(int i = 0; i < length; i++) {;
		// adding accumulated flow to flow of subsequence
		Mat incr = Mat::zeros(height, width, CV_64FC2);
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				// pixel lies in occlusion
				if(occluded.at<uchar>(y,x) == 1) {
					continue;
				}

				Vec2d f_corr = acc_flow.at<Vec2d>(y,x) + Vec2d(y,x);	// compute pixel in frame i

				if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
					int r_y = max(0, min(height - 1,(int) round(f_corr[0])));
					int r_x = max(0, min(width - 1,(int) round(f_corr[1])));

					// pixel becomes occluded
					if(occlusions[start + i].at<uchar>(r_y, r_x) != 0) {
						occluded.at<uchar>(y,x) = 1;
						continue;
					}

					Vec2d vec = Vec2d(bilinearInterp(f_corr[1],f_corr[0],forward[start + i][0]->data, forward[start + i][0]->height, forward[start + i][0]->width, forward[start + i][0]->stride),
									  bilinearInterp(f_corr[1],f_corr[0],forward[start + i][1]->data, forward[start + i][1]->height, forward[start + i][1]->width, forward[start + i][1]->stride));	// bilinear interpolation

					incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + vec;
				}
			}
		}

		acc_flow = incr;
	}

	return occluded;
}

float accumulateFlow(Mat *acc_forward, const Mat *forward, const Mat &occlusions, uint32_t FF) {
    int height = forward[0].rows;
    int width = forward[0].cols;

    float max_rad = 0;

    Mat acc_flow = Mat::zeros(height, width, CV_64FC2);
    Vec2d last_vec;

    for(u_int32_t i = 0; i < FF; i++) {
        acc_forward[i] = Mat::zeros(height, width, CV_64FC2);

        // adding accumulated flow to flow of subsequence
        Mat incr = Mat::zeros(height, width, CV_64FC2);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
            	// pixel lies in occlusion
                if(occlusions.at<uchar>(y, x) != 0) {
                    acc_forward[i].at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                    continue;
                }

                Vec2d f_corr = acc_flow.at<Vec2d>(y,x) + Vec2d(y,x);	// compute pixel in frame i

                if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
                	last_vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],0),
                					  bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],1));	// bilinear interpolation

                    incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + last_vec;

                    f_corr = incr.at<Vec2d>(y,x) + Vec2d(y,x);	// compute pixel in frame i+1

					float rad = sqrt(incr.at<Vec2d>(y,x)[0] * incr.at<Vec2d>(y,x)[0] + incr.at<Vec2d>(y,x)[1] * incr.at<Vec2d>(y,x)[1]);
					max_rad = max(max_rad, rad);
                } else {;
                    incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + last_vec;		// CONSTANT VELOCITY
                }

                // overwrite flow
                acc_forward[i].at<Vec2d>(y,x) = incr.at<Vec2d>(y,x);
            }
        }

        acc_flow = incr;
    }

    return max_rad;
}

float accumulateFlow(Mat *acc_forward, const Mat *forward, const Mat *occlusions, uint32_t FF) {
    int height = forward[0].rows;
    int width = forward[0].cols;

    float max_rad = 0;

    Mat acc_flow = Mat::zeros(height, width, CV_64FC2);
    Mat occluded = Mat::zeros(height, width, CV_8SC1);

    for(u_int32_t i = 0; i < FF; i++) {
        acc_forward[i] = Mat::zeros(height, width, CV_64FC2);

        // adding accumulated flow to flow of subsequence
        Mat incr = Mat::zeros(height, width, CV_64FC2);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
            	// pixel lies in occlusion
                if(occluded.at<uchar>(y,x) == 1) {
                    acc_forward[i].at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                    continue;
                }

                Vec2d f_corr = acc_flow.at<Vec2d>(y,x) + Vec2d(y,x);	// compute pixel in frame i

                if(f_corr[0] >= 0 && f_corr[0] < height && f_corr[1] >= 0 && f_corr[1] < width) {
                	int r_y = max(0, min(height - 1,(int) round(f_corr[0])));
                	int r_x = max(0, min(width - 1,(int) round(f_corr[1])));

                	// pixel becomes occluded
                    if(i > 0 && occlusions[i-1].at<uchar>(r_y, r_x) != 0) {
                        acc_forward[i].at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                        occluded.at<uchar>(y,x) = 1;
                        continue;
                    }

                	Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],0),
                					  bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],1));	// bilinear interpolation

                    incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + vec;

                    float rad = sqrt(incr.at<Vec2d>(y,x)[0] * incr.at<Vec2d>(y,x)[0] + incr.at<Vec2d>(y,x)[1] * incr.at<Vec2d>(y,x)[1]);
                    max_rad = max(max_rad, rad);
                } else {
                	 incr.at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                	 occluded.at<uchar>(y,x) = 1;
                }

                // overwrite flow
                acc_forward[i].at<Vec2d>(y,x) = incr.at<Vec2d>(y,x);
            }
        }

        acc_flow = incr;
    }

    return max_rad;
}

float accumulateBatches(Mat *acc_forward, Mat *acc_backward, const Mat *forward, const Mat *backward, const Mat& mask, uint32_t FF, int S, uint32_t skip, int threads) {
    int oheight = forward[0].rows;
    int owidth = forward[0].cols;

    float max_rad = 0;

    int xy_incr = skip + 1;
    int xy_start = 0.5f * skip;
	int height = floor((1.0f * oheight) / xy_incr);
	int width = floor((1.0f * owidth) / xy_incr);

    Mat acc_flow = Mat::zeros(height, width, CV_64FC2);
    Mat acc_flow_back = Mat::zeros(height, width, CV_64FC2);

    Mat last_flow = Mat::zeros(height, width, CV_64FC2);
    Mat last_flow_back = Mat::zeros(height, width, CV_64FC2);

    for(u_int32_t i = 0; i < FF; i++) {
        acc_forward[i] = Mat::zeros(height, width, CV_64FC2);
        if(acc_backward != NULL) acc_backward[(FF - 1) - i] = Mat::zeros(height, width, CV_64FC2);

        int jet_frame = (i % (S-1));

        // adding accumulated flow to flow of subsequence
        Mat incr = Mat::zeros(height, width, CV_64FC2);
        Mat incr_back = Mat::zeros(height, width, CV_64FC2);
		#pragma omp parallel for num_threads(threads)
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(mask.at<double>(y,x) == 0){
                    acc_forward[i].at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                    if(acc_backward != NULL) acc_backward[(FF - 1) - i].at<Vec2d>(y,x) = Vec2d(UNKNOWN_FLOW,UNKNOWN_FLOW);
                    continue;
                }

                Vec2d f_corr = acc_flow.at<Vec2d>(y,x) + Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);	// compute pixel in frame i

                if(f_corr[0] >= 0 && f_corr[0] < oheight && f_corr[1] >= 0 && f_corr[1] < owidth) {
                	Vec2d vec = Vec2d(bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],0),
                					  bilinearInterp<double>(f_corr[1],f_corr[0],forward[i],1));	// bilinear interpolation

                    incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + vec;

                    if(jet_frame == 0) last_flow.at<Vec2d>(y,x) = vec;

                    float rad = sqrt(incr.at<Vec2d>(y,x)[0] * incr.at<Vec2d>(y,x)[0] + incr.at<Vec2d>(y,x)[1] * incr.at<Vec2d>(y,x)[1]);
                    max_rad = max(max_rad, rad);
                } else
                	 incr.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + (jet_frame + 1) * last_flow.at<Vec2d>(y,x); 			//  Constant Velocity!!!!!!!!!!

                if(acc_backward != NULL) {
					Vec2d b_corr = acc_flow_back.at<Vec2d>(y,x) + Vec2d(y * xy_incr + xy_start, x * xy_incr + xy_start);	// compute pixel in frame (F-2) - i

					if(b_corr[0] >= 0 && b_corr[0] < oheight && b_corr[1] >= 0 && b_corr[1] < owidth) {
						Vec2d vec = Vec2d(bilinearInterp<double>(b_corr[1],b_corr[0],backward[(FF - 1) - i],0),
										  bilinearInterp<double>(b_corr[1],b_corr[0],backward[(FF - 1) - i],1));	// bilinear interpolation

						incr_back.at<Vec2d>(y,x) = acc_flow_back.at<Vec2d>(y,x) + vec;

						if(jet_frame == 0) last_flow_back.at<Vec2d>(y,x) = vec;
					} else
						incr_back.at<Vec2d>(y,x) = acc_flow.at<Vec2d>(y,x) + (jet_frame + 1) * last_flow_back.at<Vec2d>(y,x); 	//  Constant Velocity!!!!!!!!!!
                }

                // overwrite flow
                acc_forward[i].at<Vec2d>(y,x) = incr.at<Vec2d>(y,x);
                if(acc_backward != NULL) acc_backward[(FF - 1) - i].at<Vec2d>(y,x) = incr_back.at<Vec2d>(y,x);
            }
        }

        // adding last flow of subsequence to accumulated flow
        if(jet_frame == (S-2)) {	// S-2 last flow
            acc_flow = incr;
            acc_flow_back = incr_back;
        }
    }

    return max_rad;
}

Mat flowColorImg(const image_t *w_x, const image_t *w_y, int verbose, float maxrad) {
    /*
     *  adapted colorcode.h from flowcode
     */
    int width = w_x->width, height = w_x->height;

    Mat f_img(height, width, CV_8UC3);

    int x, y;
    if(maxrad <= 0) {
        // determine motion range:
        double maxx = -999, maxy = -999;
        double minx =  999, miny =  999;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                double fx = w_x->data[y*w_x->stride+x];
                double fy = w_y->data[y*w_y->stride+x];
                if (fabs(fx) > width || fabs(fy) > height) continue;
                maxx = max(maxx, fx);
                maxy = max(maxy, fy);
                minx = min(minx, fx);
                miny = min(miny, fy);
                float rad = sqrt(fx * fx + fy * fy);
                maxrad = max(maxrad, rad);
            }
        }

        // DEBUG: max and min motion
        if (verbose > 0)
            printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n", maxrad, minx, maxx, miny, maxy);
    }

    if (maxrad == 0) // if flow == 0 everywhere
        maxrad = 1;

    // DEBUG: normalization factor
    if (verbose > 0) fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            double fx = w_x->data[y*w_x->stride+x];
            double fy = w_y->data[y*w_y->stride+x];
            uchar *pix = new uchar[3];
            if (std::isnan(fx) != 0 || std::isnan(fy) != 0 || fabs(fx) > width || fabs(fy) > height) {
                pix[0] = pix[1] = pix[2] = 0;
            } else {
                computeColor(fx/maxrad, fy/maxrad, pix);
            }
            f_img.at<Vec3b>(y,x) = Vec3b(pix[0],pix[1],pix[2]);

            delete[] pix;
        }
    }

    return f_img;
}

Mat flowColorImg(const Mat &flow, int verbose, float maxrad, Mat mask) {
    if(mask.empty())
        mask = Mat::ones(flow.rows, flow.cols, CV_64FC1);
    /*
     *  adapted colorcode.h from flowcode
     */
    int width = flow.cols, height = flow.rows;

    Mat f_img(height, width, CV_8UC3);

    int x, y;
    if(maxrad <= 0) {
        // determine motion range:
        double maxx = -999, maxy = -999;
        double minx =  999, miny =  999;
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                double fx = flow.at<Vec2d>(y,x)[1];
                double fy = flow.at<Vec2d>(y,x)[0];
                if (mask.at<double>(y,x) == 0 || fabs(fx) > width || fabs(fy) > height) continue;
                maxx = max(maxx, fx);
                maxy = max(maxy, fy);
                minx = min(minx, fx);
                miny = min(miny, fy);
                float rad = sqrt(fx * fx + fy * fy);
                maxrad = max(maxrad, rad);
            }
        }

        // DEBUG: max and min motion
        if (verbose > 0)
            printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n", maxrad, minx, maxx, miny, maxy);
    }

    if (maxrad == 0) // if flow == 0 everywhere
        maxrad = 1;

    // DEBUG: normalization factor
    if (verbose > 0) fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            double fx = flow.at<Vec2d>(y,x)[1];
            double fy = flow.at<Vec2d>(y,x)[0];
            uchar *pix = new uchar[3];
            if (mask.at<double>(y,x) == 0 || fabs(fx) > width || fabs(fy) > height) {
                pix[0] = pix[1] = pix[2] = 0;
            } else {
                computeColor(fx/maxrad, fy/maxrad, pix);
            }
            f_img.at<Vec3b>(y,x) = Vec3b(pix[0],pix[1],pix[2]);

            delete[] pix;
        }
    }

    return f_img;
}

void writeToFile(Mat& mat, String file) {
	fstream myFile(file.c_str(), ios::out | ios::binary);
	for (int y = 0; y < mat.rows; y++) {
		myFile.write((char*) mat.ptr(y), mat.cols * sizeof(double));// Matlab expects double
	}
	myFile.close();
}

void readFromFile(Mat& mat, String file) {
	fstream myFile(file.c_str(), ios::in | ios::binary);
	myFile.seekg (0, ios::beg);

	if (!myFile.is_open()) {
		cerr << "Error reading file " << file << "!" << endl;
		return;
	}

	if(mat.cols == 0 && mat.rows == 0 ) {
		int width, height;
		myFile.read((char*) &height, sizeof(int));	// read height
		myFile.read((char*) &width, sizeof(int));	// read width
		mat.create(height, width, CV_64FC1);
	}

	for (int y = 0; y < mat.rows && !myFile.eof(); y++) {
		myFile.read((char*) mat.ptr(y), mat.cols * sizeof(double));	// Matlab expects double
	}
	if (myFile.eof()) cerr << "Error reading " << file << "!" << endl;
	myFile.close();
}

Mat readGTMiddlebury(string filename) {
    CFloatImage img;
    ReadFlowFile(img, filename.c_str());

    // convert cfloatimage to mat
    Mat output;
    CFImgToMat(img, output);

    return output;
}

void writeFlowMiddlebury(Mat img, string filename)
{
    // convert mat to cfloatimage
    CShape sh(img.cols, img.rows, img.channels());
    CFloatImage output(sh);

    MatToCFImg(img, output);

    WriteFlowFile(output, filename.c_str());
}

/*
 * read edges from a binary file containing width*height float32 values
 *
 * 		USE ROW ORDER FOR
 */
float_image read_float_file(string filename, const int width, const int height){
    float_image res = empty_image(float, width, height);
    FILE *fid = fopen(filename.c_str(), "rb");
    assert(fread(res.pixels, sizeof(float), width*height, fid)== (uint) (width*height));
    fclose(fid);
    return res;
}

/*
 * read edges from a binary file containing width*height float32 values
 *
 * 		USE ROW ORDER FOR
 */
void write_float_file(string filename, float_image& res, int width, int height){
    FILE *fid = fopen(filename.c_str(), "wb");
    assert(fwrite(res.pixels, sizeof(float), width*height, fid) == (uint) (width*height));
    fclose(fid);
}

void bayer2rgb(Mat src, Mat dst, int red_x, int red_y) {
	// interpolate missing colors
	for(int x = 0; x < src.cols; x++) {
		for(int y = 0; y < src.rows; y++) {
			int xm1 = (x > 0) ? (x - 1) : (x + 1);
			int xp1 = (x < src.cols - 1) ? (x + 1) : (x - 1);
			int ym1 = (y > 0) ? (y - 1) : (y + 1);
			int yp1 = (y < src.rows - 1) ? (y + 1) : (y - 1);

			float src_ym1_x = src.at<float>(ym1,x);
			float src_yp1_x = src.at<float>(yp1,x);
			float src_y_xm1 = src.at<float>(y,xm1);
			float src_y_xp1 = src.at<float>(y,xp1);

			float src_ym1_xm1 = src.at<float>(ym1,xm1);
			float src_yp1_xm1 = src.at<float>(yp1,xm1);
			float src_ym1_xp1 = src.at<float>(ym1,xp1);
			float src_yp1_xp1 = src.at<float>(yp1,xp1);

			if((y + (1 - red_y)) % 2 == 0) {
				// blue row
				if((x + red_x) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[0] = 0.5 * (src_ym1_x + src_yp1_x); 	// R
					dst.at<Vec3f>(y, x)[1] = src.at<float>(y,x);				// G
					dst.at<Vec3f>(y, x)[2] = 0.5 * (src_y_xm1 + src_y_xp1); 	// B
				} else {
					// blue
					dst.at<Vec3f>(y, x)[0] = 0.25 * (src_ym1_xm1 + src_ym1_xp1 + src_yp1_xm1 + src_yp1_xp1); 	// R
					dst.at<Vec3f>(y, x)[1] = 0.25 * (src_ym1_x + src_yp1_x + src_y_xm1 + src_y_xp1); 			// G
					dst.at<Vec3f>(y, x)[2] = src.at<float>(y,x); 												// B
				}
			} else {
				// red row
				if((x + (1 - red_x)) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[0] = 0.5 * (src_y_xm1 + src_y_xp1);		// R
					dst.at<Vec3f>(y, x)[1] = src.at<float>(y,x); 				// G
					dst.at<Vec3f>(y, x)[2] = 0.5 * (src_ym1_x + src_yp1_x); 	// B
				} else {
					// red
					dst.at<Vec3f>(y, x)[0] = src.at<float>(y,x); 												// R
					dst.at<Vec3f>(y, x)[1] = 0.25 * (src_ym1_x + src_yp1_x + src_y_xm1 + src_y_xp1); 			// G
					dst.at<Vec3f>(y, x)[2] = 0.25 * (src_ym1_xm1 + src_ym1_xp1 + src_yp1_xm1 + src_yp1_xp1); 	// B
				}
			}

		}
	}
}

void bayer2rgbGR(Mat src, Mat dst, int red_x, int red_y) {
	// first interpolate green channel since denser
	for(int x = 0; x < src.cols; x++) {
		for(int y = 0; y < src.rows; y++) {
			int xm1 = (x > 0) ? (x - 1) : (x + 1);
			int xp1 = (x < src.cols - 1) ? (x + 1) : (x - 1);
			int ym1 = (y > 0) ? (y - 1) : (y + 1);
			int yp1 = (y < src.rows - 1) ? (y + 1) : (y - 1);

			float src_ym1_x = src.at<float>(ym1,x);
			float src_yp1_x = src.at<float>(yp1,x);
			float src_y_xm1 = src.at<float>(y,xm1);
			float src_y_xp1 = src.at<float>(y,xp1);

			if((y + (1 - red_y)) % 2 == 0) {
				// blue row
				if((x + red_x) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[1] = src.at<float>(y,x);
				} else {
					// blue
					dst.at<Vec3f>(y, x)[1] = 0.25 * (src_ym1_x + src_yp1_x + src_y_xm1 + src_y_xp1);
				}
			} else {
				// red row
				if((x + (1 - red_x)) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[1] = src.at<float>(y,x);
				} else {
					// red
					dst.at<Vec3f>(y, x)[1] = 0.25 * (src_ym1_x + src_yp1_x + src_y_xm1 + src_y_xp1);
				}
			}

		}
	}

	// interpolate missing values for red and blue using green ratio
	for(int x = 0; x < src.cols; x++) {
		for(int y = 0; y < src.rows; y++) {
			int xm1 = (x > 0) ? (x - 1) : (x + 1);
			int xp1 = (x < src.cols - 1) ? (x + 1) : (x - 1);
			int ym1 = (y > 0) ? (y - 1) : (y + 1);
			int yp1 = (y < src.rows - 1) ? (y + 1) : (y - 1);

			float src_ym1_x = src.at<float>(ym1,x);
			float src_yp1_x = src.at<float>(yp1,x);
			float src_y_xm1 = src.at<float>(y,xm1);
			float src_y_xp1 = src.at<float>(y,xp1);

			float src_ym1_xm1 = src.at<float>(ym1,xm1);
			float src_yp1_xm1 = src.at<float>(yp1,xm1);
			float src_ym1_xp1 = src.at<float>(ym1,xp1);
			float src_yp1_xp1 = src.at<float>(yp1,xp1);

			// green channel
			float green_ym1_x = dst.at<Vec3f>(ym1,x)[1];
			float green_yp1_x = dst.at<Vec3f>(yp1,x)[1];
			float green_y_xm1 = dst.at<Vec3f>(y,xm1)[1];
			float green_y_xp1 = dst.at<Vec3f>(y,xp1)[1];

			float green_ym1_xm1 = dst.at<Vec3f>(ym1,xm1)[1];
			float green_yp1_xm1 = dst.at<Vec3f>(yp1,xm1)[1];
			float green_ym1_xp1 = dst.at<Vec3f>(ym1,xp1)[1];
			float green_yp1_xp1 = dst.at<Vec3f>(yp1,xp1)[1];

			if((y + (1 - red_y)) % 2 == 0) {
				// blue row
				if((x + red_x) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[0] = dst.at<Vec3f>(y,x)[1] * 0.5 * (src_ym1_x / green_ym1_x + src_yp1_x / green_yp1_x); 	// R
					dst.at<Vec3f>(y, x)[2] = dst.at<Vec3f>(y,x)[1] * 0.5 * (src_y_xm1 / green_y_xm1 + src_y_xp1 / green_y_xp1); 	// B
				} else {
					// blue
					dst.at<Vec3f>(y, x)[0] = dst.at<Vec3f>(y,x)[1] * 0.25 * (src_ym1_xm1 / green_ym1_xm1 + src_ym1_xp1 / green_ym1_xp1 + src_yp1_xm1 / green_yp1_xm1 + src_yp1_xp1 / green_yp1_xp1); 	// R
					dst.at<Vec3f>(y, x)[2] = src.at<float>(y,x); 																																		// B
				}
			} else {
				// red row
				if((x + (1 - red_x)) % 2 == 0) {
					// green
					dst.at<Vec3f>(y, x)[0] = dst.at<Vec3f>(y,x)[1] * 0.5 * (src_y_xm1 / green_y_xm1 + src_y_xp1 / green_y_xp1);		// R
					dst.at<Vec3f>(y, x)[2] = dst.at<Vec3f>(y,x)[1] * 0.5 * (src_ym1_x / green_ym1_x + src_yp1_x / green_yp1_x); 	// B
				} else {
					// red
					dst.at<Vec3f>(y, x)[0] = src.at<float>(y,x); 																																		// R
					dst.at<Vec3f>(y, x)[2] = dst.at<Vec3f>(y,x)[1] * 0.25 * (src_ym1_xm1 / green_ym1_xm1 + src_ym1_xp1 / green_ym1_xp1 + src_yp1_xm1 / green_yp1_xm1 + src_yp1_xp1 / green_yp1_xp1); 	// B
				}
			}

		}
	}
}

void rawWeighting(color_image_t* weights, int red_x, int red_y, float weight) {
	weight = fmin(fmax(weight, 0.0), 3.0);

	// set weights
	for(int x = 0; x < weights->width; x++) {
		for(int y = 0; y < weights->height; y++) {
			if((y + (1 - red_y)) % 2 == 0) {
				// blue row
				if((red_y == 1 && (x + (1 - red_x)) % 2 == 0) ||
						(red_y == 0 && (x + red_x) % 2 == 0)) {
					// green
					weights->c1[y * weights->stride + x] = 0.5 * (3 - weight); 	// R
					weights->c2[y * weights->stride + x] = weight; 				// G
					weights->c3[y * weights->stride + x] = 0.5 * (3 - weight); 	// B
				} else {
					// blue
					weights->c1[y * weights->stride + x] = 0.5 * (3 - weight); 	// R
					weights->c2[y * weights->stride + x] = 0.5 * (3 - weight); 	// G
					weights->c3[y * weights->stride + x] = weight; 				// B
				}
			} else {
				// red row
				if((red_y == 0 && (x + (1 - red_x)) % 2 == 0) ||
					(red_y == 1 && (x + red_x) % 2 == 0)) {
					// green
					weights->c1[y * weights->stride + x] = 0.5 * (3 - weight);	// R
					weights->c2[y * weights->stride + x] = weight; 				// G
					weights->c3[y * weights->stride + x] = 0.5 * (3 - weight); 	// B
				} else {
					// red
					weights->c1[y * weights->stride + x] = weight; 				// R
					weights->c2[y * weights->stride + x] = 0.5 * (3 - weight); 	// G
					weights->c3[y * weights->stride + x] = 0.5 * (3 - weight); 	// B
				}
			}

		}
	}
}
