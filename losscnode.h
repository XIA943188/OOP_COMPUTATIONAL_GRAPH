#ifndef COMPUTATIONAL_GRAPH_LOSSCNODE_H
#define COMPUTATIONAL_GRAPH_LOSSCNODE_H

#include <math.h>
#include "calcnode.h"
#include "tensor.h"

//仅支持Tensor类型的计算图实现损失函数
class LossCNode : public CalcNode<Tensor>{

};

#endif