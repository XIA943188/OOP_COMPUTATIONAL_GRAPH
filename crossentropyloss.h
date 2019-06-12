#ifndef CROSSENTROPY_LOSS_FUNCTION
#define CROSSENTROPY_LOSS_FUNCTION
#include "calcnode.h"

template <typename _T>
class CrossEntropyLoss : public CalcNode<_T>{
private:
    static const std::string DimErrMsg;
    static const std::string RankErrMsg;

protected:
	_T Calc(); //重载Calc，在这里进行计算
	_T DerCalc(Node <Tensor> *operand);

    //输入矩阵维度为ans(label), target(1)，输出矩阵维度为1
public:
    using CalcNode<_T>::Result;
    using CalcNode<_T>::DerResult;
    using CalcNode<_T>::OperandNum;
    using CalcNode<_T>::Operands;
    using CalcNode<_T>::CalcNode;
};

template <typename _T>
const std::string CrossEntropyLoss<_T>::DimErrMsg = "Dimension Mismatches.";

template <typename _T>
const std::string CrossEntropyLoss<_T>::RankErrMsg = "The rank in target is out of range.";

template <>
Tensor CrossEntropyLoss<Tensor>::Calc(){
    Tensor softmax = Operands[0]->GetVal().softmax();
    Tensor target = Operands[1]->GetVal();
    if(softmax.shape().size() != 1 || target.shape().size() != 1 || target.shape_size(0) != 1)
        throw DimErrMsg;
    Tensor output(Shape({1}), 0.0f);
    if(target.elem(0) < 0 || target.elem(0) >= softmax.shape_size(0))
        throw RankErrMsg;
    output.elem(0) = - 1 * log(softmax.elem(target.elem(0)));
    return output;
}

template <>
Tensor CrossEntropyLoss<Tensor>::DerCalc(Node<Tensor> * operand){
    Tensor der;
    if(this == operand)
        der = Tensor(Shape({1,1}), 1.0);
    else{
        Tensor der = Operands[0]->GetVal().softmax();
        Tensor target = Operands[1]->GetVal();
        if(der.shape().size() != 1 || target.shape().size() != 1 || target.shape_size(0) != 1)
            throw DimErrMsg;
        if(target.elem(0) < 0 || target.elem(0) >= der.shape_size(0))
            throw RankErrMsg;
        der.elem(target.elem(0))--;
        der = der * Operands[0]->GetDer(operand);
    }
    DerResult = new Tensor(der);
    return *DerResult;
}
#endif