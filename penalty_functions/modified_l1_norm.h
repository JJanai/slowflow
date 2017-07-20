#ifndef MODIFIEDL1NORM_H
#define MODIFIEDL1NORM_H

#include "penalty_function.h"

class ModifiedL1Norm : public PenaltyFunction
{
	const v4sf one = { 1, 1, 1, 1 };
	const v4sf two = { 2, 2, 2, 2 };

public:
    ModifiedL1Norm(float e = 0.001) :
        epsilon_sq(e*e) {
    	epsilon_vec_sq[0] = epsilon_sq;
    	epsilon_vec_sq[1] = epsilon_sq;
    	epsilon_vec_sq[2] = epsilon_sq;
    	epsilon_vec_sq[3] = epsilon_sq;
    }

    inline float apply(float xsq) {
        return sqrt(xsq + epsilon_sq);
    }

    inline v4sf apply(v4sf xsq) {
        return __builtin_ia32_sqrtps(xsq + epsilon_vec_sq);
    }

    inline float derivative(float xsq) {
        return 1/(2*sqrt(xsq + epsilon_sq));
    }

    inline v4sf derivative(v4sf xsq) {
        return one / (two * __builtin_ia32_sqrtps(xsq + epsilon_vec_sq));
    }

private:
    double epsilon_sq;
    v4sf epsilon_vec_sq;
};

#endif // MODIFIEDL1NORM_H
