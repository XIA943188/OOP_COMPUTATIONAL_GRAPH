#ifndef CROSSENTROPY_LOSS_FUNCTION
#define CROSSENTROPY_LOSS_FUNCTION
#include "losscnode.h"

class CrossEntropyLoss : public LossCNode{
private:
    static const std::string DimErrMsg;
    static const std::string RankErrMsg;

    int target{-1};

protected:
	Tensor Calc(); //重载Calc，在这里进行计算
	Tensor DerCalc(Node <Tensor> *operand);

public:
    void SetTarget(const int& _target){
        target = _target;
    }
    //输入矩阵维度为ans(label)，输出矩阵维度为1，target为int变量
};

const std::string CrossEntropyLoss::DimErrMsg = "Dimension Mismatches.";
const std::string CrossEntropyLoss::RankErrMsg = "The rank in target is out of range.";

Tensor CrossEntropyLoss::Calc(){
    Tensor softmax = Operands[0]->GetVal().softmax();
    if(softmax.shape().size() != 1)
        throw DimErrMsg;
    Tensor output(Shape({1}), 0.0f);
    if(target < 0 || target >= softmax.shape_size(0))
        throw RankErrMsg;
    output.elem(0) = - 1 * log(softmax.elem(target));
    return output;
}

Tensor CrossEntropyLoss::DerCalc(Node<Tensor> * operand){
    Tensor der;
    if(this == operand)
        der = Tensor(Shape({1,1}), 1.0);
    else{
        Tensor softmax = Operands[0]->GetVal().softmax();
        if(target < 0 || target >= softmax.shape_size(0))
            throw RankErrMsg;
        der = Tensor(softmax);
        der.elem(target)--;
        der = der * Operands[0]->GetDer(operand);//此处是否有bug？
    }
    return der;
}
#endif