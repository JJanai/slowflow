#ifndef PENALTY_FUNCTION_H
#define PENALTY_FUNCTION_H

#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
typedef __v4sf v4sf;

class PenaltyFunction
{
public:
    PenaltyFunction() {}
    virtual ~PenaltyFunction() {}

    virtual float apply(float xsq) = 0;
    virtual v4sf apply(v4sf xsq) = 0;

    virtual float derivative(float xsq) = 0;
    virtual v4sf derivative(v4sf xsq) = 0;
};

#endif // PENALTY_FUNCTION_H
