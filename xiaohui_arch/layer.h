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
    
    LSTensorPtr<T> grad_inp;
    LSTensorPtr<T> grad_out;
    
    int mx_size_;

public:
    Layer(int mx_size, std::string name):
        LayerBase<T>(name),
        mx_size_(mx_size),
        inp(new LSTensor<T>(mx_size, FixedMemory)), 
        out(new LSTensor<T>(mx_size, FixedMemory)),
        grad_inp(new LSTensor<T>(mx_size, FixedMemory)), 
        grad_out(new LSTensor<T>(mx_size, FixedMemory)), 
        op_a(mx_size, "op_a"),
        op_b(mx_size, "op_b"),
        op_c(mx_size, "op_c"),
        op_d(mx_size, "op_d") {
            LayerBase<T>::LayerVec.push_back(this); // 将layer压入全局队列中
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
        LayerBase<T>::construct_life_cycle(); // 计算生命周期，全局只执行一次。
        inp->set_tensor(inp_ptr);
        out->set_tensor(out_ptr);
        
        /*
            A = inp + 1
            B = inp + 10
            C = inp + 100
            D = B + C
        */

        op_a.Forward(1, size);
        op_b.Forward(10, size);
        op_c.Forward(100, size);
        op_d.Forward(size);
    }

    void Backward(int size, const T* grad_out_ptr, T* grad_inp_ptr) {
        LayerBase<T>::construct_life_cycle(); // 计算生命周期，全局只执行一次。
        grad_out->set_tensor(grad_out_ptr);
        grad_inp->set_tensor(grad_inp_ptr);

        op_d.Backward(size);
        op_c.Backward(size);
        op_b.Backward(size);
        op_a.Backward(size);

    }
};
