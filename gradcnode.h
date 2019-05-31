#ifndef COMPUTATIONAL_GRAPH_GRADCNODE_H
#define COMPUTATIONAL_GRAPH_GRADCNODE_H

#include"calcnode.h"

template<typename _T>
class GradCNode : public CalcNode<_T>
{
protected:
    _T Calc();
    _T DerCalc(Node <_T> *operand);
public:
    using CalcNode<_T>::Result;
	using CalcNode<_T>::DerResult;
    using CalcNode<_T>::OperandNum;
    using CalcNode<_T>::Operands;
    using CalcNode<_T>::CalcNode;
    using CalcNode<_T>::DerCalc;
};

template<typename _T>
_T GradCNode<_T>::Calc()
{
    throw "ERROR: Cannot derivate with GRAD";
}

template<>
double GradCNode<double>::DerCalc(Node <double> *operand)
{
    double der = Operands[0]->GetDer(operand);
	DerResult = new double(der);
    return *DerResult;
}


#endif //COMPUTATIONAL_GRAPH_GRADCNODE_H