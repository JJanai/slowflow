/*
 * hypothesis.h
 *
 *  Created on: Jan 18, 2016
 *      Author: Janai
 */

#ifndef HYPOTHESIS_H_
#define HYPOTHESIS_H_

#include <stdio.h>
#include <iostream>

#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "../penalty_functions/penalty_function.h"

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10
#define UNKNOWN_FLOW_THRESH 1e9

using namespace std;
using namespace cv;

enum EXTRA_TYPE { LINEAR_EXTRAPOLATION = 0, QUADRATIC_EXTRAPOLATION = 1 };
enum COMP_METHOD {ADJ = 0, ACC = 1, FINAL = 2};

class hypothesis {
public:
	hypothesis():
		jet_est(0), energy(DBL_MAX), p(Point2d(0,0)), flow_x(NULL), flow_y(NULL), occlusions(NULL), F(0), startF(0), endF(0), not_extrapolated_length(0), extrapolation_err(0) {
	}

	hypothesis(int F_):
		jet_est(0), energy(DBL_MAX), p(Point2d(0,0)), flow_x(new double[F_]), flow_y(new double[F_]), occlusions(NULL), F(F_), startF(0), endF(F_), not_extrapolated_length(F_), extrapolation_err(0) {
	}

	hypothesis(int F_, int startF_, int endF_) :
		jet_est(0), energy(DBL_MAX), p(Point2d(0,0)), flow_x(new double[F_]), flow_y(new double[F_]), occlusions(NULL), F(F_), startF(startF_), endF(endF_), not_extrapolated_length(endF_ - startF_), extrapolation_err(0) {
	}

	hypothesis(int F_, double *f_x, double *f_y, double x, double y) :
		jet_est(0), energy(DBL_MAX), p(Point2d(x,y)), flow_x(f_x), flow_y(f_y), occlusions(NULL), F(F_), startF(0), endF(F_), not_extrapolated_length(F_), extrapolation_err(0) {
	}

	hypothesis(int F_, Vec2d f_, Vec2d p_) :
		jet_est(0), energy(DBL_MAX), p(Point2d(p_[1],p_[0])), flow_x(new double[F_]), flow_y(new double[F_]), occlusions(NULL), F(F_), startF(0), endF(F_), not_extrapolated_length(F_), extrapolation_err(0) {

		for(int t = 0; t < F; t++) {
			flow_y[t] = f_[0] * (t + 1);
			flow_x[t] = f_[1] * (t + 1);
		}
	}

	hypothesis(int F_, int startF_, int endF_, double *f_x, double *f_y, double x, double y) :
		jet_est(0), energy(DBL_MAX), p(Point2d(x,y)), flow_x(f_x), flow_y(f_y), occlusions(NULL),  F(F_), startF(startF_), endF(endF_), not_extrapolated_length(endF_ - startF_), extrapolation_err(0) {
	}

	hypothesis(const hypothesis& h) :
		jet_est(h.jet_est), energy(h.energy), p(h.p), flow_x(NULL), flow_y(NULL), occlusions(NULL), F(h.F), startF(h.startF), endF(h.endF), not_extrapolated_length(h.not_extrapolated_length), extrapolation_err(h.extrapolation_err) {
		if(h.flow_x != NULL) {
			flow_x = new double[F];
			memcpy(flow_x, h.flow_x, sizeof(double) * F);
		}
		if(h.flow_y != NULL) {
			flow_y = new double[F];
			memcpy(flow_y, h.flow_y, sizeof(double) * F);
		}
		if(h.occlusions != NULL) {
			occlusions = new int[F + 1];
			memcpy(occlusions, h.occlusions, sizeof(int) * (F + 1));
		}
	}

	hypothesis(const hypothesis* h) :
		jet_est(h->jet_est), energy(h->energy), p(h->p), flow_x(NULL), flow_y(NULL), occlusions(NULL), F(h->F), startF(h->startF), endF(h->endF), not_extrapolated_length(h->not_extrapolated_length), extrapolation_err(h->extrapolation_err) {
		if(h->flow_x != NULL) {
			flow_x = new double[F];
			memcpy(flow_x, h->flow_x, sizeof(double) * F);
		}
		if(h->flow_y != NULL) {
			flow_y = new double[F];
			memcpy(flow_y, h->flow_y, sizeof(double) * F);
		}
		if(h->occlusions != NULL) {
			occlusions = new int[(F + 1)];
			memcpy(occlusions, h->occlusions, sizeof(int) * (F + 1));
		}
	}

	~hypothesis() {
		if(flow_x != NULL) delete[] flow_x;
		if(flow_y != NULL) delete[] flow_y;
		if(occlusions != NULL) delete[] occlusions;
	}

	// extrapolate hypothesis to finalize
	hypothesis* new_complete(int approach);

	hypothesis* new_perturbed(double u_p, double v_p);

	void setLocation(double x_, double y_) {
		p.x = x_;
		p.y = y_;
	}

	void setOcclusions(const Mat* forward_flow, const Mat* backward_flow, float occlusion_threshold = 5.0, float occlusion_fb_threshold = 10.0);

	static hypothesis* outlier(int F, PenaltyFunction* phi, float e = 1.0f, int cmp_F_ = 0, int cmp_fps_ = 1, int jet_jps_ = 1) {
		if(cmp_F_ == 0) cmp_F_ = F;

		hypothesis* o = new hypothesis(F, 0, F);
		o->jet_est = 0;
		o->occlusions = new int[F+1];
		o->occlusions[0] = 0;
		for(int t = 0; t < F; t++) {
			o->flow_y[t] = UNKNOWN_FLOW;
			o->flow_x[t] = UNKNOWN_FLOW;
			o->occlusions[t+1] = 0;
		}
		o->energy = phi->apply(e * e);
		return o;
	}

	void replaceFlow(double *f_x, double *f_y) {
		if(flow_x != NULL) delete[] flow_x;
		if(flow_y != NULL) delete[] flow_y;
		flow_x = f_x;
		flow_y = f_y;
	}

	void adaptFPS(int nF) {
		if(F > nF && F % nF != 0) cerr << "WARNING: " << F << " is not a multiple of " << nF << "!" << endl;
		if(nF > F && nF % F != 0) cerr << "WARNING: " << nF << " is not a multiple of " << F << "!" << endl;
		float skip = (1.0f * F) / nF;

		F = nF;

		startF = 0;
		endF = F;
		double* n_flow_x = new double[F];
		double* n_flow_y = new double[F];

		if(skip >= 1) {
			for(int i = 0; i < F; i++) {
				int off = i * skip + (skip - 1);
				n_flow_x[i] = flow_x[off];
				n_flow_y[i] = flow_y[off];
			}
		} else {
			for(int i = 0; i < F; i++) {
				float last_x = 0;
				float last_y = 0;

				int off = floor(i * skip);
				int offm1 = floor((i - 1) * skip);

				if(i > 0) {
					last_x = flow_x[offm1];
					last_y = flow_y[offm1];
				}

				n_flow_x[i] = last_x + skip * (flow_x[off] - last_x);	// scale last flow
				n_flow_y[i] = last_y + skip * (flow_y[off] - last_y);	// scale last flow

			}
		}

		delete[] flow_x;
		delete[] flow_y;
		delete[] occlusions;
		flow_x = n_flow_x;
		flow_y = n_flow_y;
	}

	hypothesis& operator =(const hypothesis& h);

	double getEnergy() {
		return energy;
	}

	// compare and check status
	double distance(hypothesis& h, int method = ACC);
	/*
	 * output:
	 * 		-2: distance > thres
	 * 		-1: this hypothesis shorter
	 * 		 0: both similar
	 * 		 1: this hypothesis longer
	 */
	int compare(const hypothesis& h, double thres, int method);
	int compare(int x, int y, const Mat *acc_cons_flow, int start, int end, double thres=0.1, int method = ACC);

	bool completed() const {
		return (startF == 0 && endF == F);
	}

	// get pixels and flow
	double x(int t = 0) {
		if(t <= 0) return p.x;

		return p.x + flow_x[t - 1];
	}

	double y(int t = 0) {
		if(t <= 0) return p.y;

		return p.y + flow_y[t - 1];
	}

	double u(int t = 0) const {
		return flow_x[t];
	}

	double v(int t = 0) const {
		return flow_y[t];
	}

	int occluded(int t = 0) {
		if(occlusions == NULL)
			return 0;
		else
			return occlusions[t];
	}

	int visible (int t = 0) {
		return 1 - occluded(t);
	}

	// get first frame
	int start() {
		return startF;
	}

	int extrapolation_length() {
		return F - not_extrapolated_length;
	}

	int extrapolation_error() {
		return extrapolation_err;
	}

	float score() {
		return energy;
	}

	int jet_est;
	double energy;
	Point2d p;
	double* flow_x,* flow_y;
	int* occlusions;
private:
	int F;
	int startF, endF;

	int not_extrapolated_length;
	double extrapolation_err;
};

#endif /* HYPOTHESIS_H_ */
