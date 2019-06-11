#ifndef CROSSENTROPY_LOSS_FUNCTION
#define CROSSENTROPY_LOSS_FUNCTION
#include "loss.h"

class CrossEntropyLoss : public LossFunction{
private:
    static const std::string DimErrMsg;
    static const std::string RankErrMsg;
public:
    Tensor operator() (const Tensor& ans, const Tensor& target);//输入矩阵维度为ans(bat*dim), target(dim)，输出矩阵维度为dim
};

const std::string CrossEntropyLoss::DimErrMsg = "Dimension Mismatched.";
const std::string CrossEntropyLoss::RankErrMsg = "The rank in target out of range.";

Tensor CrossEntropyLoss::operator() (const Tensor& ans, const Tensor& target){
    if(ans.dim() != 2 || target.dim() != 1 || ans.shape_size(0) != target.shape_size(0))
        throw DimErrMsg;
    Tensor output(target.shape(), 0.0f);
    for(int i = 0; i < output.shape_size(0); i++){
        int rank = int(target.elem(i));
        if(rank < 0 || rank >= ans.shape_size(1))
            throw RankErrMsg;
        output.elem(i) = - 1 * log(ans.elem(Shape({i, rank})));
    }
    return output;
}

#endif