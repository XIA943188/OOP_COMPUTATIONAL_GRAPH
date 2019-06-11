#ifndef COMPUTATIONAL_GRAPH_MINCNODE_H
#define COMPUTATIONAL_GRAPH_MINCNODE_H

#include"calcnode.h"

template<typename _T>
class MinCNode : public CalcNode<_T>
{
protected:
    _T Calc(); //重载Calc，在这里进行计算
    _T DerCalc(Node <_T> *operand);

public:
    using CalcNode<_T>::Result;
    using CalcNode<_T>::DerResult;
    using CalcNode<_T>::OperandNum;
    using CalcNode<_T>::Operands;
    using CalcNode<_T>::CalcNode;
};

template<>
double MinCNode<double>::Calc()
{
    Result = new double(Operands[0]->GetVal() - Operands[1]->GetVal());
    return *Result;
}

template<>
Tensor MinCNode<Tensor>::Calc()
{
    Result = new Tensor(Operands[0]->GetVal() - Operands[1]->GetVal());
    return *Result;
}

template<>
double MinCNode<double>::DerCalc(Node <double> *operand)
{
    double der = (this == operand) ? 1.0 : Operands[0]->GetDer(operand) - Operands[1]->GetDer(operand);
    DerResult = new double(der);
    return *DerResult;
}

template<>
Tensor MinCNode<Tensor>::DerCalc(Node <Tensor> *operand)
{
    Tensor der = (this == operand) ? Tensor(Shape({1, 1}), 1.0) : Operands[0]->GetDer(operand) - Operands[1]->GetDer(operand);
    DerResult = new Tensor(der);
    return *DerResult;
}

#endif //COMPUTATIONAL_GRAPH_MINCNODE_H
