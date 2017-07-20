#ifndef TRUNCMODIFIEDL1NORM_H
#define TRUNCMODIFIEDL1NORM_H

#include "penalty_function.h"

class TruncModifiedL1Norm : public PenaltyFunction
{
	const v4sf one = { 1, 1, 1, 1 };
	const v4sf two = { 2, 2, 2, 2 };

public:
	TruncModifiedL1Norm(float e = 0.001, float trunc = 0.5) :
		epsilon_sq(e*e), truncation(trunc)  {
    	epsilon_vec_sq[0] = epsilon_sq;
    	epsilon_vec_sq[1] = epsilon_sq;
    	epsilon_vec_sq[2] = epsilon_sq;
    	epsilon_vec_sq[3] = epsilon_sq;
    }

    inline float apply(float xsq) {
        if(sqrt(xsq) > truncation)
			return sqrt(truncation + epsilon_sq);
		else
			return sqrt(xsq + epsilon_sq);
    }

    inline v4sf apply(v4sf xsq) {
        v4sf out = __builtin_ia32_sqrtps(xsq + epsilon_vec_sq);

		if(sqrt((float) xsq[0]) > truncation) out[0] = sqrt(truncation + epsilon_sq);
		if(sqrt((float) xsq[1]) > truncation) out[1] = sqrt(truncation + epsilon_sq);
		if(sqrt((float) xsq[2]) > truncation) out[2] = sqrt(truncation + epsilon_sq);
		if(sqrt((float) xsq[3]) > truncation) out[3] = sqrt(truncation + epsilon_sq);

		return out;
    }

    inline float derivative(float xsq) {
        if(sqrt(xsq) > truncation)
			return 0;
		else
			return 1 / (2 * sqrt(xsq + epsilon_sq));
    }

    inline v4sf derivative(v4sf xsq) {
    	v4sf out = one / (two * __builtin_ia32_sqrtps(xsq + epsilon_vec_sq));

		if(sqrt((float) xsq[0]) > truncation) out[0] = 0;
		if(sqrt((float) xsq[1]) > truncation) out[1] = 0;
		if(sqrt((float) xsq[2]) > truncation) out[2] = 0;
		if(sqrt((float) xsq[3]) > truncation) out[3] = 0;

		return out;
    }

private:
    float epsilon_sq;
    v4sf epsilon_vec_sq;
    float truncation;
};

#endif // TRUNCMODIFIEDL1NORM_H
