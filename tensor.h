#ifndef COMPUTATIONAL_GRAPH_TENSOR_H
#define COMPUTATIONAL_GRAPH_TENSOR_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>

namespace EXPAND_TYPE {
    const int NOT_CAP = -1;
    const int MATCH = 0;
    const int OP1_EXP_ROW = 1;
    const int OP1_EXP_COL = 2;
    const int OP2_EXP_ROW = 4;
    const int OP2_EXP_COL = 8;
};

using namespace std;
using namespace EXPAND_TYPE;

typedef pair<int, int> Shape;

class Tensor {
    vector<double> _elem;
    int _row_num; //行数
    int _col_num; //列数
    static const string ErrMsg;
public:
    Tensor() {}
    Tensor(const int &row_num_, const int &col_num_, const vector<double> &elem_): _row_num(row_num_), _col_num(col_num_), _elem(elem_) {}
    Tensor(const int &row_num_, const int &col_num_, const double &val_ = 0.0): _row_num(row_num_), _col_num(col_num_) {
        vector<double> elem_ = vector<double>(size());
        for (auto it : elem_) it = val_;
        _elem = elem_;
    }
    Tensor(const Tensor &t) {
        int new_size = t.size();
        vector<double> new_elem(new_size); 
        for (int i = 0; i < new_size; i++) new_elem[i] = t.elem(i);
        _elem = new_elem; _row_num = t.row(); _col_num = t.col();
    }

    Shape shape() const { return make_pair(_row_num, _col_num); }
    int size() const { return _row_num * _col_num; }
    int row() const { return _row_num; }
    int col() const { return _col_num; }

    double &elem(const int &row, const int &col) { return _elem[row * _col_num + col]; }
    double &elem(const int &rank) { return _elem[rank]; }
    double elem(const int &row, const int &col) const { return _elem[row * _col_num + col]; }
    double elem(const int &rank) const { return _elem[rank]; }

    bool row_cap(const Tensor &t) { return (row() == t.row()); }
    bool col_cap(const Tensor &t) { return (col() == t.col()); }
    bool mul_cap(const Tensor &t) { return (col() == t.row()); }
    bool size_cap(const Shape &shape) { return (size() == shape.first * shape.second); }
    int broadcast_cap(const Tensor &t);
    
    Tensor &transpose();
    Tensor &reshape(const Shape &new_shape);
    Tensor broadcast_sum(const Tensor &t);
    Tensor broadcast_min(const Tensor &t);
    Tensor broadcast_mul(const Tensor &t);
    Tensor concat(const Tensor &t, const int &op_dim);

    Tensor operator+(const Tensor &t);
    Tensor operator-() const ;
    Tensor operator-(const Tensor &t);
    Tensor operator*(const Tensor &t);
    Tensor operator*(const double &d);

    double norm();
    int argmax(); //返回元素最大的rank

	friend ostream &operator<<(ostream &out, const Tensor &t) {
        out << "[";
		for (int _row = 0; _row < t._row_num; _row++) {
			out << "[";
			for (int _col = 0; _col < t._col_num - 1; _col++)
				out << t.elem(_row, _col) << ", ";
			out << t.elem(_row, t._col_num - 1) << "]";
            if (_row < t._row_num - 1) out << ",\n";
		}
        out << "]\n";
		return out;
	}

    friend istream &operator>>(istream &in, Tensor &t) { //默认输入格式为[ [ 0 , 1 ] , [ 1 , 2 ] ]
        string cmd, input;
        getline(in, cmd);
        istringstream ist(cmd);
        int row_num = 0, size_num = 0;
        vector<double> new_elem;
        ist >> input;
        while (ist >> input) {
            if (input == "," || input == "]")
                continue;
            else if (input == "[") //如果是"["则行数增加
                row_num++;
            else {
                double val = atof(input.c_str());
                size_num++;
                new_elem.push_back(val);
            }
        }
        t = Tensor(row_num, size_num / row_num, new_elem);
        return in;
    }
};

const string Tensor::ErrMsg = "ERROR: shape of tensor incompatible";

int Tensor::broadcast_cap(const Tensor &t) { //似乎没有必要，只需要判断相容即可
    int exp_type = MATCH;
    if (row() != t.row()) {
        if (row() == 1) exp_type += OP1_EXP_ROW;
        else if (t.row() == 1) exp_type += OP2_EXP_ROW;
        else return NOT_CAP;
    }
    if (col() != t.col()) {
        if (col() == 1) exp_type += OP1_EXP_COL;
        else if (t.col() == 1) exp_type += OP2_EXP_COL;
        else return NOT_CAP;
    }
    return exp_type;
}

Tensor &Tensor::transpose() {
    if (_row_num == 1 || _col_num == 1) swap(_row_num, _col_num);
    else {
        vector<double> new_elem = vector<double>(size());
        for (int _row = 0; _row < _col_num; _row++)
            for (int _col = 0; _col < _row_num; _col++)
                new_elem[_row * _row_num + _col] = elem(_col, _row);
        _elem = new_elem;
        swap(_row_num, _col_num);
    }
    return *this;
}

Tensor &Tensor::reshape(const Shape &new_shape){
    if (!size_cap(new_shape)) throw ErrMsg;
    _row_num = new_shape.first; _col_num = new_shape.second;
    return *this;
}

Tensor Tensor::broadcast_sum(const Tensor &t) {
    int exp_type = broadcast_cap(t);
    if (exp_type < 0) throw ErrMsg;
    int row_iter = (exp_type ^ 1) ? t.row() : row(); //只需要找最大值
    int col_iter = (exp_type >> 1 ^ 1) ? t.col() : col(); //只需要找最大值
    vector<double> new_elem = vector<double>(row_iter * col_iter);
    for (int _row = 0; _row < row_iter; _row++) 
        for (int _col = 0; _col < col_iter; _col++) 
            new_elem[_row * col_iter + _col] = elem(_row % row(), _col % col()) + t.elem(_row % t.row(), _col % t.col());
    Tensor sum = Tensor(row_iter, col_iter, new_elem);
    return sum;
}

Tensor Tensor::broadcast_min(const Tensor &t) {
    return broadcast_sum(-t);
}

Tensor Tensor::broadcast_mul(const Tensor &t) {
    int exp_type = broadcast_cap(t);
    if (exp_type < 0) throw ErrMsg;
    int row_iter = (exp_type ^ 1) ? t.row() : row(); //只需要找最大值
    int col_iter = (exp_type >> 1 ^ 1) ? t.col() : col(); //只需要找最大值
    vector<double> new_elem = vector<double>(row_iter * col_iter);
    for (int _row = 0; _row < row_iter; _row++) 
        for (int _col = 0; _col < col_iter; _col++) 
            new_elem[_row * col_iter + _col] = elem(_row % row(), _col % col()) * t.elem(_row % t.row(), _col % t.col());
    Tensor prod = Tensor(row_iter, col_iter, new_elem);
    return prod;
}

//0代表左右相拼，1代表上下相拼
Tensor Tensor::concat(const Tensor &t, const int &op_dim) {
    Tensor res;
    if (op_dim) {
        if (!col_cap(t)) throw ErrMsg;
        int row_num = row() + t.row(), col_num = col();
        vector<double> new_elem = vector<double>(row_num * col_num);
        for (int _row = 0; _row < _row_num; _row++)
            for (int _col = 0; _col < col_num; _col++)
                new_elem[_row * col_num + _col] = elem(_row, _col);
        for (int _row = _row_num; _row < row_num; _row++)
            for (int _col = 0; _col < col_num; _col++)
                new_elem[_row * col_num + _col] = t.elem(_row - _row_num, _col);
        res = Tensor(row_num, col_num, new_elem);
    }
    else {
        if (!row_cap(t)) throw ErrMsg;
        int row_num = row(), col_num = col() + t.col();
        vector<double> new_elem = vector<double>(row_num * col_num);
        for (int _row = 0; _row < row_num; _row++) {
            for (int _col = 0; _col < _col_num; _col++)
                new_elem[_row * col_num + _col] = elem(_row, _col);
            for (int _col = _col_num; _col < col_num; _col++)
                new_elem[_row * col_num + _col] = t.elem(_row, _col - _col_num);
        }
        res = Tensor(row_num, col_num, new_elem);
    }
    return res;
}

Tensor Tensor::operator+(const Tensor &t) { //按正常的矩阵加法，也可以调用broadcast_sum
    return broadcast_sum(t);
}

Tensor Tensor::operator-() const {
    for (auto it : _elem) it = -it;
    return *this;
}

Tensor Tensor::operator-(const Tensor &t) {
    return *this + (-t);
}

Tensor Tensor::operator*(const Tensor &t) {
    if (!mul_cap(t)) throw ErrMsg;
    int row_num = row(), col_num = t.col();
    vector<double> new_elem = vector<double>(row_num * col_num);
    for (int _row = 0; _row < row_num; _row++) {
        for (int _col = 0; _col < col_num; _col++) {
            double val = 0;
            for (int i = 0; i < col(); i++)
                val += elem(_row, i) * t.elem(i, _col);
        }
    }
    Tensor prod = Tensor(row_num, col_num, new_elem);
    return prod;    
}

Tensor Tensor::operator*(const double &d)
{
    return broadcast_mul(Tensor(1, 1, d));
}

int Tensor::argmax() {
    double max_elem = elem(0); int max_rank = 0;
    for (int i = 1; i < size(); i++) 
        if (elem(i) > max_elem) {
            max_elem = elem(i); max_rank = i;
        }
    return max_rank;
}

double Tensor::norm() {
    double res = 0.0;
    for (auto it : _elem) res += it * it;
    return sqrt(res);
}




#endif