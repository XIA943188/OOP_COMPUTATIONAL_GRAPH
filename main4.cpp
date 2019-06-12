#include <iostream>
#include <sstream>
#include "comgraph.h"
#include "basic_calc_pack.h"
#include "advanced_calc_pack.h"
#include "compare_calc_pack.h"
#include "tensor.h"

typedef string string;
using namespace std;
ostream &ErrOut = cout; //应项目要求，错误信息向cout输出
ostream &PriOut = cout; //应项目要求，PriNode信息向cout输出
ostream &AnsOut = cout; //答案输出至cout

//默认只支持Tensor版本，double类型则需要转化为1维Tensor
//测试样例说明：
//节点x：类型Placeholder，维度为10*1的Tensor
//节点W：类型Variable，维度为5*10的Tensor
//节点b：类型Variable，维度为5*1的Tensor
//节点y：类型Placeholder，维度5*1的Tensor
//节点loss：类型为MSELoss，target为5*1的Tensor（相当于label）
//优化任务：Wx+b与target的交叉熵损失

const int iter_num = 10;

int main() {
	ComGraph<Tensor> neural_network(ErrOut, PriOut); 
	//神经网络的建立-------BEGIN-------
    neural_network.BuildPHNode("x");
	neural_network.BuildVarNode("W", Tensor(Shape({5, 10}), 0.0));
    neural_network.BuildVarNode("b", Tensor(Shape({5, 1}), 1.0));
    neural_network.BuildPHNode("y");
    neural_network.BuildCalcNode<MulCNode<Tensor>>("mul", 2, vector<string>({"W", "x"}));
    neural_network.BuildCalcNode<PluCNode<Tensor>>("plu", 2, vector<string>({"mul", "b"}));
    neural_network.BuildCalcNode<MSELoss<Tensor>>("loss", 2, vector<string>({"plu", "y"}));
    neural_network.BuildCalcNode<GradCNode<Tensor>>("grad", 1, vector<string>({"loss"}));
    neural_network.BuildCalcNode<DerCNode<Tensor>>("dW", 2, vector<string>({"grad", "W"}));
    neural_network.BuildCalcNode<DerCNode<Tensor>>("db", 2, vector<string>({"grad", "b"}));
    //神经网络的建立-------END-------
    //优化过程开始--------BEGIN------
    auto sample_x = Tensor(Shape({10, 1}), Elem({1, 2, 3, 4, 5, 4, 3, 2, 1, 0}));
    auto label_y = Tensor(Shape({5, 1}), Elem({0, 1, 0, 0, 0}));
    vector<pair<string, Tensor>> PHList;
    PHList.push_back(make_pair("x", sample_x)); PHList.push_back(make_pair("y", label_y));
    for (int iter = 0; iter < iter_num; iter++) {
        bool failed = false; Tensor Res;
        AnsOut << "# iter: " << iter << endl;
        try {
            Res = neural_network.Eval("loss", PHList);
        }
        catch (string &ErrMsg) {
            ErrOut << ErrMsg << endl;
            failed = true;
        }
        
        if (!failed) {
            auto W = neural_network.Eval("W", PHList), b = neural_network.Eval("b", PHList);
            AnsOut << "W:\n" << W << "b:\n" << b << "loss: " << Res << endl;
        }
        
        try {
            neural_network.GradientDescend("dW", "W", PHList);
            neural_network.GradientDescend("db", "b", PHList);
        }
        catch (string &ErrMsg) {
            ErrOut << ErrMsg << endl;
        }
        
    }
    //优化过程结束--------END------
    return 0;
}

