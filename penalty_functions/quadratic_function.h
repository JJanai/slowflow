#ifndef QUADRATIC_FUNCTION_H
#define QUADRATIC_FUNCTION_H

#include "penalty_function.h"

class QuadraticFunction : public PenaltyFunction
{
	const v4sf one = { 1, 1, 1, 1 };

public:
    QuadraticFunction() {
    }

    inline float apply(float xsq) {
        return xsq;
    }

    inline v4sf apply(v4sf xsq) {
        return xsq;
    }

    inline float derivative(float xsq) {
        return 1;
    }

    inline v4sf derivative(v4sf xsq) {
        return one;
    }
};

#endif // QUADRATIC_FUNCTION_H
