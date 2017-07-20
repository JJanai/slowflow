#ifndef GEMANMCCLURE_H
#define GEMANMCCLURE_H

#include "penalty_function.h"

class GemanMcClure : public PenaltyFunction
{
	const v4sf one = { 1, 1, 1, 1 };
	const v4sf two = { 2, 2, 2, 2 };
public:
	GemanMcClure(float e = 0.05) :
    	epsilon_sq(e*e) {
    	epsilon_vec_sq[0] = epsilon_sq;
    	epsilon_vec_sq[1] = epsilon_sq;
    	epsilon_vec_sq[2] = epsilon_sq;
    	epsilon_vec_sq[3] = epsilon_sq;

    }

    inline float apply(float xsq) {
        return xsq / ((xsq + 1) *(xsq + 1));
    }

    inline v4sf apply(v4sf xsq) {
        return xsq / ((xsq + one) *(xsq + one));
    }

    inline float derivative(float xsq) {
    	float tmp = (epsilon_sq + xsq);
    	tmp = tmp * tmp;
        return (epsilon_sq + 2 * xsq) / tmp;
    }

    inline v4sf derivative(v4sf xsq) {
    	v4sf tmp = (epsilon_vec_sq + xsq);
    	tmp = tmp * tmp;
        return (epsilon_vec_sq + two * xsq) / tmp;
    }

private:
    double epsilon_sq;
    v4sf epsilon_vec_sq;
};

#endif // GEMANMCCLURE_H
