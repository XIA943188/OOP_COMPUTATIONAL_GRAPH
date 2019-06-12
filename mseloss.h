#ifndef MSE_LOSS_FUNCTION
#define MSE_LOSS_FUNCTION
#include "losscnode.h"

class MSELoss : public LossCNode{
private:
    static const std::string DimErrMsg;

protected:
    Tensor Calc(); //重载Calc，在这里进行计算
	Tensor DerCalc(Node <Tensor> *operand);

    //输入矩阵维度为ans(dim), target(dim)，输出维度为1
};

const std::string MSELoss::DimErrMsg = "Dimension Mismatched.";

Tensor MSELoss::Calc(){
    Tensor ans = Operands[0]->GetVal();
    Tensor target = Operands[1]->GetVal();
    if(ans.dim() != 1 || target.dim() != 1 || ans.shape_size(0) != target.shape_size(0))
        throw DimErrMsg;
    Tensor output(Shape({1}), 0.0f);
    for(int i = 0; i < ans.shape_size(0); i++){
        double a = ans.elem(i), b = target.elem(i);
        output.elem(i) += (a-b) * (a-b);
    }
    output.elem(0) /= ans.shape_size(0);
    return output;
}

Tensor MSELoss::DerCalc(Node <Tensor> *operand){
    Tensor der;
    if(this == operand)
        der = Tensor(Shape({1,1}), 1.0);
    else{
        der = Operands[0]->GetVal();
        Tensor target = Operands[1]->GetVal();
        if(der.dim() != 1 || target.dim() != 1 || der.shape_size(0) != target.shape_size(0))
            throw DimErrMsg;
        der = der - target;
        der = der * ( 2.0 / double(der.shape_size(0)) );
        der = der * Operands[0]->GetDer(operand);//此处是否有bug？
    }
    DerResult = new Tensor(der);
    return *DerResult;
}
#endif