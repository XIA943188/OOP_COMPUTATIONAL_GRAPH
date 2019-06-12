#ifndef COMPUTATIONAL_GRAPH_TENSOR_H
#define COMPUTATIONAL_GRAPH_TENSOR_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

typedef std::vector<int> Shape;
typedef std::vector<double> Elem;

int shape2size(const Shape &s) { //将shape转化成size
    int total_size = 1;
    for (auto it : s) total_size *= it;
        return total_size;
}

Shape rank2shape_rank(const int &rank, const Shape &shape) { //将循环量rank转化成每个维度上的分量
    int _dim = shape.size();
    Shape shape_rank(_dim);
    int tmp = rank;
    for (int d = _dim - 1; d >= 0; d--) { //得到对应在每个维度上的分量
        shape_rank[d] = tmp % shape[d];
        tmp /= shape[d];
    }
    return shape_rank;
}

int shape_rank2rank(const Shape &shape_rank, const Shape &shape) { //将每个维度的分量转化成循环量rank，加上取模操作方便broadcast操作，所以使用时需要额外注意
    int _dim = shape.size(), rank = 0;
    for (int d = 1; d < _dim; d++) {
        rank += (shape_rank[d - 1] % shape[d - 1]);
        rank *= shape[d];
    }
    rank += (shape_rank[_dim - 1] % shape[_dim - 1]);
    return rank;
}

int rank2count(const int &rank, const Shape &shape) {
    auto shape_rank = rank2shape_rank(rank, shape);
    int dim = shape_rank.size(), count = 0;
    for (int d = dim - 1; d >= 0; d--)
        if (shape_rank[d] == 0) count++;
        else break;
    return count;
}

class Tensor {
    Elem _elem;
    int _dim; //维数
    Shape _shape; //各个维度上的尺寸
    static const std::string ErrMsg;
public:
    Tensor() {}
    Tensor(const Shape &shape_, const Elem &elem_): _dim(shape_.size()), _shape(shape_), _elem(elem_) {} //通过shape和elem构造
    Tensor(const Shape &shape_, Elem &&elem_): _dim(shape_.size()), _shape(shape_) { _elem = std::move(elem_); }
    Tensor(const Shape &shape_, const double &val_):  _dim(shape_.size()), _shape(shape_) { //默认所有元素均为d，形状为shape（只在标量乘法用到）
        _elem = Elem(size());
        for (auto it = _elem.begin(); it != _elem.end(); it++) *it = val_;
    }
    Tensor(const Elem &elem_): _dim(1), _elem(elem_) { //默认构造一个1*n的Tensor
        _shape = Shape(1); _shape[0] = elem_.size();
    }
    Tensor(Elem &&elem_): _dim(1) { //默认构造一个1*n的Tensor
        _shape = Shape(1); _shape[0] = elem_.size();
        _elem = std::move(elem_);
    }
    Tensor(const Tensor &t) {
        int new_size = t.size();
        Elem new_elem(new_size); 
        for (int i = 0; i < new_size; i++) new_elem[i] = t.elem(i);
        _elem = new_elem; _shape = t.shape(); _dim = t.dim();
    }

    int size() const { return shape2size(_shape); } //返回元素个数
    int dim() const { return _dim; } 
    Shape shape() const { return _shape; }
    int shape_size(const int &dim) const { return _shape[dim]; } //返回dim维度的尺寸

    double &elem(const int &rank) { return _elem[rank]; }
    double elem(const int &rank) const { return _elem[rank]; }
    double &elem(const Shape &shape_rank) { return elem(shape_rank2rank(shape_rank, _shape)); }
    double elem(const Shape &shape_rank) const { return elem(shape_rank2rank(shape_rank, _shape)); }

    bool size_cap(const Shape &s) { return (size() == shape2size(s)); } //比较元素个数是否相容
    bool broadcast_cap(const Tensor &t) const; //比较是否能够broadcast
    bool shape_cap(const Tensor &t, const int &dim) const { return (shape_size(dim) == t.shape_size(dim)); } //比较在dim维度是否相容，用于concat

    Tensor &transpose();
    Tensor &reshape(const Shape &new_shape);
    Tensor broadcast_sum(const Tensor &t) const;
    Tensor broadcast_min(const Tensor &t) const;
    Tensor broadcast_mul(const Tensor &t) const; //broadcast版本的乘法
    Tensor broadcast_div(const Tensor &t) const;
    Tensor concat(const Tensor &t, const int &op_dim) const;
    Tensor relu() const;
    Tensor der_relu() const;
    Tensor softmax() const;
    Tensor sqrt() const;
    Tensor reduce_sum(const int &op_dim) const;
    Tensor reduce_mul(const int &op_dim) const;

    double norm() const;
    int argmax() const; //返回元素最大的rank

    Tensor operator+(const Tensor &t);
    Tensor operator-() const ;
    Tensor operator-(const Tensor &t);
    Tensor operator*(const Tensor &t); //矩阵乘法，默认计算2维
    Tensor operator*(const double &d); //每个元素乘一个标量

    friend std::ostream &operator<<(std::ostream &out, const Tensor &t) {
        int size_ = t.size(), dim_ = t.dim(); bool left_flag = true;
        for (int rank = 0; rank <= size_; rank++) {
            int count = rank2count(rank, t._shape);
            if (count == dim_) //需要输出左括号
                if (left_flag) {
                    while (count--) out << "[";
                    left_flag = false;
                    out << t.elem(rank);
                }
                else
                    while (count--) out << "]";
            else {
                int count_ = count;
                while (count_--) out << "]";
                if (count >= dim_ - 1 && count) out << "\n";
                else out << " ";
                while (count--) out << "[";
                out << t.elem(rank);
            }
        }
        out << "\n";
	return out;
    }
};

const std::string Tensor::ErrMsg = "ERROR: shape of tensor incompatible";

bool Tensor::broadcast_cap(const Tensor &t) const {
    bool cap = true;
    for (int d = 0; d < _dim; d++)
        if (shape_size(d) != 1 && t.shape_size(d) != 1 && shape_size(d) != t.shape_size(d))
            return !cap;
    return cap;
}

//只针对二维Tensor
Tensor &Tensor::transpose() {
    if (_dim != 2) throw ErrMsg;
    if (shape_size(0) == 1 || shape_size(1) == 1) std::swap(_shape[0], _shape[1]);
    else {
        Elem new_elem(size());
        for (int row = 0; row < _shape[1]; row++)
            for (int col = 0; col < _shape[0]; col++)
                new_elem[row * _shape[0] + col] = elem(col * _shape[1] + row);
        _elem = new_elem;
        std::swap(_shape[0], _shape[1]);
    }
    return *this;
}

Tensor &Tensor::reshape(const Shape &new_shape){
    if (!size_cap(new_shape)) throw ErrMsg;
    _dim = new_shape.size(); //更新维度
    _shape.resize(_dim);
    for (int d = 0; d < _dim; d++)
        _shape[d] = new_shape[d]; //更新每一维尺寸
    return *this;
}

Tensor Tensor::broadcast_sum(const Tensor &t) const {
    if (!broadcast_cap(t)) throw ErrMsg;
    Shape new_shape(_dim), shape_rank(_dim);
    for (int d = 0; d < _dim; d++) new_shape[d] = std::max(shape_size(d), t.shape_size(d)); //判断可以broadcast后只需要取最大即可
    int new_size = shape2size(new_shape);
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++) {
        shape_rank = rank2shape_rank(rank, new_shape); //取出对应的每个维度的分量
        new_elem[rank] = elem(shape_rank) + t.elem(shape_rank); //这里按之前所说，模两个Tensor在每个维度的尺寸，即可做到broadcast，所以在shape_rank2rank中采用了取模操作
    }
    Tensor sum = Tensor(new_shape, new_elem);
    return sum;
}

Tensor Tensor::broadcast_min(const Tensor &t) const {
    return broadcast_sum(-t);
}

Tensor Tensor::broadcast_mul(const Tensor &t) const {
    if (!broadcast_cap(t)) throw ErrMsg;
    Shape new_shape(_dim), shape_rank(_dim);
    for (int d = 0; d < _dim; d++) new_shape[d] = std::max(shape_size(d), t.shape_size(d)); //判断可以broadcast后只需要取最大即可
    int new_size = shape2size(new_shape);
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++) {
        shape_rank = rank2shape_rank(rank, new_shape); //取出对应的每个维度的分量
        new_elem[rank] = elem(shape_rank) * t.elem(shape_rank); //这里按之前所说，模两个Tensor在每个维度的尺寸，即可做到broadcast，所以在shape_rank2rank中采用了取模操作
    }
    Tensor prod = Tensor(new_shape, new_elem);
    return prod;
}

Tensor Tensor::broadcast_div(const Tensor &t) const {
    if (!broadcast_cap(t)) throw ErrMsg;
    Shape new_shape(_dim), shape_rank(_dim);
    for (int d = 0; d < _dim; d++) new_shape[d] = std::max(shape_size(d), t.shape_size(d)); //判断可以broadcast后只需要取最大即可
    int new_size = shape2size(new_shape);
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++) {
        shape_rank = rank2shape_rank(rank, new_shape); //取出对应的每个维度的分量
        new_elem[rank] = elem(shape_rank) / t.elem(shape_rank); //这里按之前所说，模两个Tensor在每个维度的尺寸，即可做到broadcast，所以在shape_rank2rank中采用了取模操作
    }
    Tensor prod = Tensor(new_shape, new_elem);
    return prod;
}

Tensor Tensor::concat(const Tensor &t, const int &op_dim) const { //大致思路是：通过新Tensor当前rank对应的各维度分量，如果在op_dim分量小于this在op_dim的尺寸，就应该是this的元素，否则是t的
    for (int d = 0; d < _dim; d++)
        if (d != op_dim && !shape_cap(t, d)) //判断其余维度的相容性
            throw ErrMsg;
    Shape new_shape = _shape, shape_rank(_dim);
    new_shape[op_dim] += t.shape_size(op_dim);
    int new_size = shape2size(new_shape);
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++) {
        shape_rank = rank2shape_rank(rank, new_shape);
        if (shape_rank[op_dim] < shape_size(op_dim))
            new_elem[rank] = elem(shape_rank);
        else {
            shape_rank[op_dim] -= shape_size(op_dim);
            new_elem[rank] = t.elem(shape_rank);
        }
    }
    return Tensor(new_shape, new_elem);
}

Tensor Tensor::relu() const {
    int new_size = size();
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++)
        new_elem[rank] = std::max(0.0, _elem[rank]);
    return Tensor(shape(), new_elem);
}

Tensor Tensor::der_relu() const {
    int new_size = size();
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++)
        new_elem[rank] = (_elem[rank] > 0) ? 1.0 : 0.0;
    return Tensor(shape(), new_elem);
}

Tensor Tensor::softmax() const {
    int new_size = size();
    Elem new_elem(new_size);
    double sum = 0.0;
    for (auto it : _elem) sum += exp(it);
    for (int rank = 0; rank < new_size; rank++)
        new_elem[rank] = _elem[rank] / sum;
    return Tensor(shape(), new_elem);
}

Tensor Tensor::sqrt() const {
    int new_size = size();
    Elem new_elem(new_size);
    for (int rank = 0; rank < new_size; rank++)
        new_elem[rank] = std::sqrt(_elem[rank]);
    return Tensor(shape(), new_elem);
}

Tensor Tensor::reduce_sum(const int &op_dim) const {
    auto new_shape = _shape; new_shape[op_dim] = 1;
    auto old_size = size(), new_size = shape2size(new_shape);
    auto sum = Tensor(new_shape, 0.0);
    for (int rank = 0; rank < old_size; rank++) {
        auto shape_rank = rank2shape_rank(rank, _shape);
        sum.elem(shape_rank) += elem(rank); //这里实际上会做一次模的操作，参看shape_rank2rank
    }
    new_shape.erase(new_shape.begin() + op_dim);
    sum.reshape(new_shape);
    return sum;
}

Tensor Tensor::reduce_mul(const int &op_dim) const {
    auto new_shape = _shape; new_shape[op_dim] = 1;
    auto old_size = size(), new_size = shape2size(new_shape);
    auto prod = Tensor(new_shape, 1.0);
    for (int rank = 0; rank < old_size; rank++) {
        auto shape_rank = rank2shape_rank(rank, _shape);
        prod.elem(shape_rank) *= elem(rank); //这里实际上会做一次模的操作，参看shape_rank2rank
    }
    new_shape.erase(new_shape.begin() + op_dim);
    prod.reshape(new_shape);
    return prod;
}

double Tensor::norm() const {
    double res = 0.0;
    for (auto it : _elem) res += it * it;
    return std::sqrt(res);
}

int Tensor::argmax() const {
    double max_elem = elem(0); int max_rank = 0;
    int total_size = size();
    for (int i = 1; i < total_size; i++) 
        if (elem(i) > max_elem) {
            max_elem = elem(i); max_rank = i;
        }
    return max_rank;
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
    if (dim() != 2 || t.dim() != 2 || shape_size(1) != t.shape_size(0)) throw ErrMsg;
    Shape new_shape(2); new_shape[0] = shape_size(0); new_shape[1] = t.shape_size(1); 
    int new_size = shape2size(new_shape);
    Elem new_elem(new_size);
    for (int row = 0; row < new_shape[0]; row++) {
        for (int col = 0; col < new_shape[1]; col++) {
            double val = 0.0;
            for (int i = 0; i < shape_size(1); i++)
                val += elem(row * shape_size(1) + i) * t.elem(i * t.shape_size(1) + col);
            new_elem[row * new_shape[1] + col] = val;
        }
    }
    Tensor prod = Tensor(new_shape, new_elem);
    return prod;
}

Tensor Tensor::operator*(const double &d) {
    Shape new_shape(dim()); for (auto it = new_shape.begin(); it != new_shape.end(); it++) *it = 1;
    return broadcast_mul(Tensor(new_shape, d));
}




#endif
