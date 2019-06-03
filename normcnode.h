#ifndef COMPUTATIONAL_GRAPH_NORMCNODE_H
#define COMPUTATIONAL_GRAPH_NORMCNODE_H

#include"calcnode.h"

template<typename _T>
class NormCNode : public CalcNode<_T>
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
double NormCNode<double>::Calc()
{
	Result = new double(Operands[0]->GetVal() * Operands[0]->GetVal());
	return *Result;
}

template<>
Tensor NormCNode<Tensor>::Calc()
{
    double res = Operands[0]->GetVal().norm();
	Result = new Tensor(1, 1, res);
	return *Result;
}

template<>
double NormCNode<double>::DerCalc(Node <double> *operand)
{
	double der = (this == operand) ? 1.0 : 2 * Operands[0]->GetVal() * Operands[0]->GetDer(operand);
	DerResult = new double(der);
	return *DerResult;
}

template<>
Tensor NormCNode<Tensor>::DerCalc(Node <Tensor> *operand)
{
    Tensor der = (this == operand) ? Tensor(1, 1, 1.0) 
		: Operands[0]->GetVal().broadcast_mul(Operands[0]->GetDer(operand)) * 2;
	DerResult = new Tensor(der);
    return *DerResult;
}

#endif //COMPUTATIONAL_GRAPH_MULCNODE_H