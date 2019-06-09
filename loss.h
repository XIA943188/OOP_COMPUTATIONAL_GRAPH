#ifndef LOSS_FUNCTION
#define LOSS_FUNCTION

#include <iostream>
#include <math.h>
#include "tensor.h"

class LossFunction{
public:
    virtual Tensor operator() (const Tensor& ans, const Tensor& target) = 0;
};

#endif