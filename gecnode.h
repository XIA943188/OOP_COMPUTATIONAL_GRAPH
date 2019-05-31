#ifndef COMPUTATIONAL_GRAPH_GECNODE_H
#define COMPUTATIONAL_GRAPH_GECNODE_H

#include"calcnode.h"

template<typename _T>
class GECNode : public CalcNode<_T> //比较运算符GECNode
{
protected:
    _T Calc(); //重载Calc，在这里进行计算
    _T DerCalc(Node <_T> *operand);
public:
    using CalcNode<_T>::Result;
    using CalcNode<_T>::DerResult;
    using CalcNode<_T>::OperandNum;
    using CalcNode<_T>::Operands;           //Using 基类的操作元
    using CalcNode<_T>::CalcNode;
};

template<>
double GECNode<double>::Calc()
{
    Result = new double(Operands[0]->GetVal() >= Operands[1]->GetVal());
    return *Result;
}

template<typename _T>
_T GECNode<_T>::DerCalc(Node <_T> *operand)
{
    throw "ERROR: Cannot derivate with >=";
}

#endif //COMPUTATIONAL_GRAPH_GECNODE_H


