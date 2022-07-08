#pragma once
#include "manager.h"
#include "vector"
#include "iostream"
#include "base_class.h"
#include "operator.h"

template<class T>
class Layer: public LayerBase<T> {
private:
    OperatorIncrease<T> op_a;
    OperatorIncrease<T> op_b;
    OperatorIncrease<T> op_c;
    OperatorAdd<T> op_d;

    LSTensorPtr<T> inp;
    LSTensorPtr<T> out;
    
    int mx_size_;

public:
    Layer(int mx_size, std::string name):
        LayerBase<T>(name),
        mx_size_(mx_size),
        inp(new LSTensor<T>(mx_size, FixedMemory)), 
        out(new LSTensor<T>(mx_size, FixedMemory)),
        op_a(mx_size, "op_a"),
        op_b(mx_size, "op_b"),
        op_c(mx_size, "op_c"),
        op_d(mx_size, "op_d") {
            LayerBase<T>::LayerVec.push_back(this);
        }

    void init_forward() {
        op_d.set_out(out);
        op_a.init_forward(inp);

        op_b.init_forward(op_a.out());
        op_c.init_forward(op_b.out());
        op_d.init_forward(op_b.out(), op_c.out());
    }

    void init_backward() {

    }

    void Forward(int size, const T* inp_ptr, T* out_ptr) {
        LayerBase<T>::construct_life_cycle();
        inp->set_tensor(inp_ptr);
        out->set_tensor(out_ptr);

        op_a.Forward(1, size);
        op_b.Forward(10, size);
        op_c.Forward(100, size);
        op_d.Forward(size);
    }

    void Backward() {
        
    }
};
