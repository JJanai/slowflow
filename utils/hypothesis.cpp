/*
 * hypothesis.cpp
 *
 *  Created on: Feb 8, 2016
 *      Author: Janai
 */

#include "hypothesis.h"
#include "utils.h"


hypothesis* hypothesis::new_complete(int approach) {
	int l = endF - startF + 1;

	hypothesis* h = NULL;

	if(approach == LINEAR_EXTRAPOLATION) {
		if(l < 2) return NULL;

		double t[l], x[l], y[l];
		// get coordinates for fitting
		t[0] = startF;							// frame starting from start
		x[0] = p.x;
		y[0] = p.y;
		for(int f = 0; f < (l - 1); f++) {
			t[f + 1] = startF + f + 1;
			x[f + 1] = p.x + flow_x[startF + f];
			y[f + 1] = p.y + flow_y[startF + f];
		}

		// fit linear function
		double y_c0, y_c1, y_cov00, y_cov01, y_cov11, y_sumsq;
		double x_c0, x_c1, x_cov00, x_cov01, x_cov11, x_sumsq;
		gsl_fit_linear(t, 1, y, 1, l, &y_c0, &y_c1, &y_cov00, &y_cov01, &y_cov11, &y_sumsq);
		gsl_fit_linear(t, 1, x, 1, l, &x_c0, &x_c1, &x_cov00, &x_cov01, &x_cov11, &x_sumsq);

		// create new hypothesis while replacing the points used to fit!
		double *new_flow_y = new double[F];
		double *new_flow_x = new double[F];
		for(int f = 0; f < F; f++) {
			new_flow_y[f] = y_c1 * (f + 1); 	// + p.y; ONLY INTERESTED IN FLOW
			new_flow_x[f] = x_c1 * (f + 1);     // + p.x;
		}

		h = new hypothesis(F, new_flow_x, new_flow_y, x_c0, y_c0);
		h->jet_est = jet_est;
	} else if(approach == QUADRATIC_EXTRAPOLATION) {
		if(l < 3) return NULL;

		double chisq_x, chisq_y;
		gsl_matrix *T, *cov_x, *cov_y;
		gsl_vector *x, *y, *w, *c_x, *c_y;

		T = gsl_matrix_alloc(l, 3);
		x = gsl_vector_alloc(l);
		y = gsl_vector_alloc(l);
		w = gsl_vector_alloc(l);

		c_x = gsl_vector_alloc(3);
		c_y = gsl_vector_alloc(3);
		cov_x = gsl_matrix_alloc(3, 3);
		cov_y = gsl_matrix_alloc(3, 3);

		int t0 = startF;							// frame starting from start
		gsl_matrix_set(T, 0, 0, 1.0);
		gsl_matrix_set(T, 0, 1, t0);
		gsl_matrix_set(T, 0, 2, t0 * t0);
		gsl_vector_set(x, 0, p.x);
		gsl_vector_set(y, 0, p.y);
		gsl_vector_set(w, 0, 1.0 / l); 				// TODO CHANGE WEIGHT

		for (int i = 0; i < l - 1; i++) {
			int ti = (t0 + i + 1);

			gsl_matrix_set(T, i + 1, 0, 1.0);
			gsl_matrix_set(T, i + 1, 1, ti);
			gsl_matrix_set(T, i + 1, 2, ti * ti);
			gsl_vector_set(x, i + 1, p.x + flow_x[startF + i]);
			gsl_vector_set(y, i + 1, p.y + flow_y[startF + i]);
			gsl_vector_set(w, i + 1, 1.0 / l);
		}

		{
			gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(l, 3);
			gsl_multifit_wlinear(T, w, x, c_x, cov_x, &chisq_x, work);
			gsl_multifit_linear_free(work);

			work = gsl_multifit_linear_alloc(l, 3);
			gsl_multifit_wlinear(T, w, y, c_y, cov_y, &chisq_y, work);
			gsl_multifit_linear_free(work);
		}

#define C_X(i) (gsl_vector_get(c_x,(i)))
#define C_Y(i) (gsl_vector_get(c_y,(i)))
//#define COV_X(i,j) (gsl_matrix_get(cov_x,(i),(j)))
//#define COV_Y(i,j) (gsl_matrix_get(cov_y,(i),(j)))

//		{
//			printf("# best fit: Y = %g + %g X + %g X^2\n", C_X(0), C_X(1), C_X(2));
//			printf("# best fit: Y = %g + %g X + %g X^2\n", C_Y(0), C_Y(1), C_Y(2));
//
//			printf("# covariance matrix:\n");
//			printf("[ %+.5e, %+.5e, %+.5e  \n", COV_X(0, 0), COV_X(0, 1), COV_X(0, 2));
//			printf("  %+.5e, %+.5e, %+.5e  \n", COV_X(1, 0), COV_X(1, 1), COV_X(1, 2));
//			printf("  %+.5e, %+.5e, %+.5e ]\n", COV_X(2, 0), COV_X(2, 1), COV_X(2, 2));
//			printf("# chisq = %g\n", chisq_x);
//		}


		// create new hypothesis while replacing the points used to fit!
		double *new_flow_y = new double[F];
		double *new_flow_x = new double[F];
		for(int f = 0; f < F; f++) {
			new_flow_x[f] = C_X(1) * (f + 1) + C_X(2) * (f + 1) * (f + 1);     	// + p.x;
			new_flow_y[f] = C_Y(1) * (f + 1) + C_Y(2) * (f + 1) * (f + 1); 		// + p.y; ONLY INTERESTED IN FLOW
		}

		h = new hypothesis(F, new_flow_x, new_flow_y, C_X(0), C_Y(0));
		h->jet_est = jet_est;

		gsl_matrix_free(T);
		gsl_vector_free(x);
		gsl_vector_free(y);
		gsl_vector_free(w);
		gsl_vector_free(c_x);
		gsl_vector_free(c_y);
		gsl_matrix_free(cov_x);
		gsl_matrix_free(cov_y);

	}

	// store extrapolation details
	h->not_extrapolated_length = l;
	h->extrapolation_err = distance(*h, ADJ);

	return h;
}


hypothesis* hypothesis::new_perturbed(double u_p, double v_p) {
	hypothesis* perturbed_h = new hypothesis(F);

	perturbed_h->jet_est = jet_est;
	perturbed_h->p = p;
	perturbed_h->not_extrapolated_length = not_extrapolated_length;
	perturbed_h->extrapolation_err = extrapolation_err;

	for(int f = 0; f < F; f++) {
		float scale = (0.9f / F) * f + 0.1;					// linear scaling

//		perturbed_h->flow_x[f] = flow_x[f] + scale * u_p;
//		perturbed_h->flow_y[f] = flow_y[f] + scale * v_p;
		if(flow_x[f] > 0)
			perturbed_h->flow_x[f] = flow_x[f] + scale * u_p;
		else
			perturbed_h->flow_x[f] = flow_x[f] - scale * u_p;

		if(flow_y[f] > 0)
			perturbed_h->flow_y[f] = flow_y[f] + scale * v_p;
		else
			perturbed_h->flow_y[f] = flow_y[f] - scale * v_p;
	}

	return perturbed_h;
}

hypothesis& hypothesis::operator =(const hypothesis& h) {
	F = h.F;

	startF = h.startF;
	endF = h.endF;

	if(this != &h) {
		flow_x = new double[h.F];
		flow_y = new double[h.F];

		memcpy(flow_x, h.flow_x, sizeof(double) * F);
		memcpy(flow_y, h.flow_y, sizeof(double) * F);
	}

	p = h.p;

	energy = h.energy;

	return *this;
}

void hypothesis::setOcclusions(const Mat* forward_flow, const Mat* backward_flow, float occlusion_threshold, float occlusion_fb_threshold) {
	if(occlusions != NULL) delete[] occlusions;
	occlusions = new int[F+1];

	occlusions[0] = 0;				// always visible in reference frame

	for(int t = 0; t < F; t++) {
		// we can not detect when something becomes visible again!
		if( occlusions[t] == 1) {
			occlusions[t+1] = 1;
			continue;
		}

		double u_tm1 = 0;
		double v_tm1 = 0;
		if (t > 0) {
			u_tm1 += flow_x[t-1];
			v_tm1 += flow_y[t-1];  // x, y
		}
		double x_tm1 = p.x + u_tm1;
		double y_tm1 = p.y + v_tm1;

		if (y_tm1 >= 0 && y_tm1 < forward_flow[t].rows && x_tm1 >= 0 && x_tm1 < forward_flow[t].cols) {
			double F_x = bilinearInterp<double>(x_tm1, y_tm1, forward_flow[t], 1);
			double F_y = bilinearInterp<double>(x_tm1, y_tm1, forward_flow[t], 0);
			double ysq = (flow_y[t] - v_tm1 - F_y);
			double xsq = (flow_x[t] - u_tm1 - F_x);

			double x_t = p.x + flow_x[t] ;
			double y_t = p.y + flow_y[t];

			if(y_t >= 0 && y_t < forward_flow[t].rows && x_t >= 0 && x_t < forward_flow[t].cols) {
				double bF_x = bilinearInterp<double>(x_t, y_t, backward_flow[t], 1);
				double bF_y = bilinearInterp<double>(x_t, y_t, backward_flow[t], 0);

				double fb_ysq = (bF_y + F_y);
				double fb_xsq = (bF_x  + F_x);

				if(sqrt(fb_ysq*fb_ysq + fb_xsq*fb_xsq) < occlusion_fb_threshold && sqrt(ysq*ysq + xsq*xsq) < occlusion_threshold)
					occlusions[t+1] = 0;
				else
					occlusions[t+1] = 1;
			} else {
				occlusions[t+1] = 1;
			}
		} else {
			occlusions[t+1] = 1;
		}
	}
}

double hypothesis::distance(hypothesis& h, int method) {
	int first = max(h.startF, startF);
	int length = min((endF - first), (h.endF - first));

	Vec2d prev_flow = Vec2d(0,0), prev_flow_h = Vec2d(0,0);
	if(first > 0) {
		if(h.startF < startF)
			prev_flow_h = Vec2d(h.flow_y[first - 1], h.flow_x[first - 1]);
		else if(h.startF > startF)
			prev_flow = Vec2d(flow_y[first - 1], flow_x[first - 1]);
	}

	double sum = 0;

	// difference of final, accumulated or adjacent flow
	if(method == FINAL) {
		// final flow
		int end = first + length;

		Vec2d flow_f = Vec2d(flow_y[end], flow_x[end]) - prev_flow;
		Vec2d flow_f_h = Vec2d(h.flow_y[end], h.flow_x[end]) - prev_flow_h;

		double ysq = (flow_f[0] - flow_f_h[0]);
		double xsq = (flow_f[1] - flow_f_h[1]);

		sum += sqrt(xsq*xsq + ysq*ysq);
	} else {
		int l = 1;
		for(int f = first; f < first + length; f++,l++) {
			double ysq = 0, xsq = 0;

			Vec2d flow_f = Vec2d(flow_y[f], flow_x[f]) - prev_flow;
			Vec2d flow_f_h = Vec2d(h.flow_y[f], h.flow_x[f]) - prev_flow_h;

			// comparison method
			if(method == ACC) {
				// accumulated flow
				ysq = flow_f[0] - flow_f_h[0];
				xsq = flow_f[1] - flow_f_h[1];

				sum += sqrt(xsq*xsq + ysq*ysq) / l;
			} else if(method == ADJ) {
				// adjacent flow
				Vec2d flow_fm1 = Vec2d(0,0), flow_fm1_h = Vec2d(0,0);

				if(f > first) {
					flow_fm1 = Vec2d(flow_y[f - 1], flow_x[f - 1]) - prev_flow;
					flow_fm1_h = Vec2d(h.flow_y[f -1], h.flow_x[f -1]) - prev_flow_h;
				}

				ysq = ((flow_f[0] - flow_fm1[0]) - (flow_f_h[0] - flow_fm1_h[0]));
				xsq = ((flow_f[1] - flow_fm1[1]) - (flow_f_h[1] - flow_fm1_h[1]));

				sum += sqrt(xsq*xsq + ysq*ysq);
			}
		}
	}

	if(method != ACC)
		sum = sum / length;	// normalize by length

	return sum;
}

//int hypothesis::compare(const hypothesis& h, double thres, int method) {
//	if(!completed() || !h.completed()) {
//		cerr << "Comparing hypotheses have different length!" << endl;
//		return -2;
//	}
//
//	int first = 0;
//	int length = F;
//
//	double dist = 0;
//	// difference of final, accumulated or adjacent flow
//	if(method == FINAL) {
//		// final flow
//		int end = first + length;
//
//		Vec2d flow_f = Vec2d(flow_y[end], flow_x[end]);
//		Vec2d flow_f_h = Vec2d(h.flow_y[end], h.flow_x[end]);
//
//		double ysq = (flow_f[0] - flow_f_h[0]);
//		double xsq = (flow_f[1] - flow_f_h[1]);
//
//		dist += sqrt(xsq*xsq + ysq*ysq);
//	} else {
//		int l = 1;
//		for(int f = first; f < first + length; f++, l++) {
//			double ysq = 0, xsq = 0;
//
//			Vec2d flow_f = Vec2d(flow_y[f], flow_x[f]);
//			Vec2d flow_f_h = Vec2d(h.flow_y[f], h.flow_x[f]);
//
//			// comparison method
//			if(method == ACC) {
//				// accumulated flow
//				ysq = flow_f[0] - flow_f_h[0];
//				xsq = flow_f[1] - flow_f_h[1];
//
//				dist += sqrt(xsq*xsq + ysq*ysq) / l;
//			} else if(method == ADJ) {
//				// adjacent flow
//				Vec2d flow_fm1 = Vec2d(0,0), flow_fm1_h = Vec2d(0,0);
//
//				if(f > first) {
//					flow_fm1 = Vec2d(flow_y[f - 1], flow_x[f - 1]);
//					flow_fm1_h = Vec2d(h.flow_y[first - 1], h.flow_x[first - 1]);
//				}
//
//				ysq = ((flow_f[0] - flow_fm1[0]) - (flow_f_h[0] - flow_fm1_h[0]));
//				xsq = ((flow_f[1] - flow_fm1[1]) - (flow_f_h[1] - flow_fm1_h[1]));
//
//				dist += sqrt(xsq*xsq + ysq*ysq);
//			}
//		}
//	}
//
////	if(method != ACC)
//		dist /= length;																		// normalize by length
//
//	if(dist > thres)   																		// compare distance to linear threshold
//		return -2;
//	else if(not_extrapolated_length < h.not_extrapolated_length)							// the other trajectory is longer
//		return -1;
//	else if(not_extrapolated_length > h.not_extrapolated_length)							// this trajectory is longer
//		return 1;
//	else if(extrapolation_err > h.extrapolation_err)										// the other has a smaller error
//		return -1;
//	else if(extrapolation_err < h.extrapolation_err)										// this trajectory has a smaller error
//		return 1;
//	else																					// both trajectories are equal
//		return 0;
//}

int hypothesis::compare(const hypothesis& h, double thres, int method) {
	int first = max(h.startF, startF);
	int length = min((endF - first), (h.endF - first));

	// flow before the start
	Vec2d prev_flow = Vec2d(0,0), prev_flow_h = Vec2d(0,0);
	if(first > 0) {
		if(h.startF < startF)
			prev_flow_h = Vec2d(h.flow_y[first - 1], h.flow_x[first - 1]);
		else if(h.startF > startF)
			prev_flow = Vec2d(flow_y[first - 1], flow_x[first - 1]);
	}

	double dist = 0;
	// difference of final, accumulated or adjacent flow
	if(method == FINAL) {
		// final flow
		int end = first + length;

		Vec2d flow_f = Vec2d(flow_y[end], flow_x[end]) - prev_flow;
		Vec2d flow_f_h = Vec2d(h.flow_y[end], h.flow_x[end]) - prev_flow_h;

		double ysq = (flow_f[0] - flow_f_h[0]);
		double xsq = (flow_f[1] - flow_f_h[1]);

		dist += sqrt(xsq*xsq + ysq*ysq);
	} else {
		int l = 1;
		for(int f = first; f < first + length; f++, l++) {
			double ysq = 0, xsq = 0;

			Vec2d flow_f = Vec2d(flow_y[f], flow_x[f]) - prev_flow;
			Vec2d flow_f_h = Vec2d(h.flow_y[f], h.flow_x[f]) - prev_flow_h;

			// comparison method
			if(method == ACC) {
				// accumulated flow
				ysq = flow_f[0] - flow_f_h[0];
				xsq = flow_f[1] - flow_f_h[1];

				dist += sqrt(xsq*xsq + ysq*ysq) / l;
			} else if(method == ADJ) {
				// adjacent flow
				Vec2d flow_fm1 = Vec2d(0,0), flow_fm1_h = Vec2d(0,0);

				if(f > first) {
					flow_fm1 = Vec2d(flow_y[f - 1], flow_x[f - 1]) - prev_flow;
					flow_fm1_h = Vec2d(h.flow_y[first - 1], h.flow_x[first - 1]) - prev_flow_h;
				}

				ysq = ((flow_f[0] - flow_fm1[0]) - (flow_f_h[0] - flow_fm1_h[0]));
				xsq = ((flow_f[1] - flow_fm1[1]) - (flow_f_h[1] - flow_fm1_h[1]));

				dist += sqrt(xsq*xsq + ysq*ysq);
			}
		}
	}

	if(method != ACC)
		dist /= length;										// normalize by length

	if(dist > thres)   							// compare distance to linear threshold
		return -2;
	else if(not_extrapolated_length < h.not_extrapolated_length)							// the other trajectory is longer
		return -1;
	else if(not_extrapolated_length > h.not_extrapolated_length)							// this trajectory is longer
		return 1;
	else if(extrapolation_err > h.extrapolation_err)										// the other has a smaller error
		return -1;
	else if(extrapolation_err < h.extrapolation_err)										// this trajectory has a smaller error
		return 1;
	else										// both trajectories are equal
		return 0;
}

int hypothesis::compare(int x, int y, const Mat *acc_cons_flow, int start, int end, double thres, int method) {
	int first = max(start, startF);
	int length = min((endF - first), (end - first));

	Vec2d prev_flow = Vec2d(0,0), prev_flow_h = Vec2d(0,0);
	if(first > 0) {
		if(start < startF)
			prev_flow_h = acc_cons_flow[first - 1].at<Vec2d>(y,x);
		else if(start > startF)
			prev_flow = Vec2d(flow_y[first - 1], flow_x[first - 1]);
	}

	double dist = 0;
	// difference of final, accumulated or adjacent flow
	if(method == FINAL) {
		// final flow
		int end = first + length;

		Vec2d flow_f = Vec2d(flow_y[end], flow_x[end]) - prev_flow;
		Vec2d flow_f_h = acc_cons_flow[end].at<Vec2d>(y,x) - prev_flow_h;

		double ysq = (flow_f[0] - flow_f_h[0]);
		double xsq = (flow_f[1] - flow_f_h[1]);

		dist += sqrt(xsq*xsq + ysq*ysq);
	} else {
		int l = 1;
		for(int f = first; f < first + length; f++, l++) {
			double ysq = 0, xsq = 0;

			Vec2d flow_f = Vec2d(flow_y[f], flow_x[f]) - prev_flow;
			Vec2d flow_f_h = acc_cons_flow[f].at<Vec2d>(y,x) - prev_flow_h;

			// comparison method
			if(method == ACC) {
				// accumulated flow
				ysq = flow_f[0] - flow_f_h[0];
				xsq = flow_f[1] - flow_f_h[1];

				dist += sqrt(xsq*xsq + ysq*ysq) / l;
			} else if(method == ADJ) {
				// adjacent flow
				Vec2d flow_fm1 = Vec2d(0,0), flow_fm1_h = Vec2d(0,0);

				if(f > first) {
					flow_fm1 = Vec2d(flow_y[f - 1], flow_x[f - 1]) - prev_flow;
					flow_fm1_h = acc_cons_flow[f - 1].at<Vec2d>(y,x) - prev_flow_h;
				}

				ysq = ((flow_f[0] - flow_fm1[0]) - (flow_f_h[0] - flow_fm1_h[0]));
				xsq = ((flow_f[1] - flow_fm1[1]) - (flow_f_h[1] - flow_fm1_h[1]));

				dist += sqrt(xsq*xsq + ysq*ysq);
			}
		}
	}

	if(method != ACC)
		dist /= length;										// normalize by length

	int diff = (endF - startF) - (end - start); // difference in length

	if(dist > thres)   							// compare distance to linear threshold
		return -2;
	else if(diff < 0)							// the other trajectory is longer
		return -1;
	else if(diff > 0)							// this trajectory is longer
		return 1;
	else										// both trajectories are equal
		return 0;
}
