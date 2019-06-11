#ifndef MSE_LOSS_FUNCTION
#define MSE_LOSS_FUNCTION
#include "loss.h"

class MSELoss : public LossFunction{
private:
    static const std::string DimErrMsg;
public:
    Tensor operator() (const Tensor& ans, const Tensor& target);//输入矩阵维度为ans(bat*dim), target(bat*dim)，输出矩阵维度为dim
};

const std::string MSELoss::DimErrMsg = "Dimension Mismatched.";

Tensor MSELoss::operator() (const Tensor& ans, const Tensor& target){
    if(ans.dim() != 2 || target.dim() != 2 || ans.shape_size(0) != target.shape_size(0) || ans.shape_size(1) != target.shape_size(1))
        throw DimErrMsg;
    Tensor output(Shape({ans.shape_size(0)}), 0.0f);
    for(int i = 0; i < output.shape_size(0); i++){
        for(int j = 0; j < ans.shape_size(1); j++){
            double a = ans.elem(Shape({i, j})), b = target.elem(Shape({i, j}));
            output.elem(i) += (a-b) * (a-b);
        }
        output.elem(i) /= ans.shape_size(1);
    }
    return output;
}

#endif