#ifndef LORENTZIAN_H
#define LORENTZIAN_H

#include "penalty_function.h"

class Lorentzian : public PenaltyFunction
{
	const v4sf one = { 1, 1, 1, 1 };
	const v4sf two = { 2, 2, 2, 2 };
public:
    Lorentzian(float e = 0.05) :
    	epsilon_sq(e*e) {
    	epsilon_vec_sq[0] = epsilon_sq;
    	epsilon_vec_sq[1] = epsilon_sq;
    	epsilon_vec_sq[2] = epsilon_sq;
    	epsilon_vec_sq[3] = epsilon_sq;

    }

    inline float apply(float xsq) {
        return log(1 + 0.5 * xsq / epsilon_sq);
    }

    inline v4sf apply(v4sf xsq) {
    	v4sf out;
    	out[0] = log(1 + 0.5 * xsq[0] / epsilon_sq);
    	out[1] = log(1 + 0.5 * xsq[1] / epsilon_sq);
    	out[2] = log(1 + 0.5 * xsq[2] / epsilon_sq);
    	out[3] = log(1 + 0.5 * xsq[3] / epsilon_sq);

        return out;
    }

    inline float derivative(float xsq) {
        return 1/(2*epsilon_sq + xsq);
    }

    inline v4sf derivative(v4sf xsq) {
        return one / (two * epsilon_vec_sq + xsq);
    }

private:
    double epsilon_sq;
    v4sf epsilon_vec_sq;
};

#endif // LORENTZIAN_H
