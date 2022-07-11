#pragma once
#include "manager.h"
template<class T>
class OperatorIncrease {
private:
    LSTensorPtr<T> inp_ = nullptr, out_ = nullptr;
    LSTensorPtr<T> inp_b = nullptr;
    int mx_size_;

    std::string name_;

public:
    OperatorIncrease(int mx_size, std::string name): 
        mx_size_(mx_size), 
        out_(new LSTensor<T>(mx_size)), 
        name_(name) {}
        
    void Forward(int val, int size) {
        T* out_tensor = out_->tensor();
        T* inp_tensor = inp_->tensor();
        for (int i = 0; i < size; i ++) {
            out_tensor[i] = inp_tensor[i] + val;
        }
    }

    // void Backward(int size) {
    //     T* grad_inp_tensor = grad_inp_->tensor();
    //     T* grad_out_tensor = grad_out_->tensor();
    //     for(int i = 0 ; i < size; i ++) {
    //         grad_inp_tensor[i] = grad_out_tensor[i];
    //     }
    // }

    void init_forward(LSTensorPtr<T> tensor_) {
        LSTensorUsage::update_operator_idx();
        inp_ = tensor_;
        inp_->update_life_idx();
        out_->update_life_idx();
    }

    // void init_backward(LSTensorPtr<T> tensor_) {
    //     LSTensorUsage::update_operator_idx();
    //     grad_out_ = tensor_;
    //     grad_out_->update_life_idx();
    //     grad_inp_->update_life_idx();
    // }

    void set_out(LSTensorPtr<T> tensor_) { 
        out_ = tensor_; 
    }
    LSTensorPtr<T> out() { return out_; }


    // void set_grad_out(LSTensorPtr<T> tensor_) { grad_out = tensor_; }
    // LSTensorPtr grad_out() { return grad_out; }
    // void set_grad_inp(LSTensorPtr<T> tensor_) { grad_inp = tensor_; }
    // LSTensorPtr grad_inp() { return grad_inp; }
    
};


template<class T>
class OperatorAdd {
private:
    LSTensorPtr<T> inp_A = nullptr, out_ = nullptr;
    LSTensorPtr<T> inp_B = nullptr;
    int mx_size_;

    std::string name_;

public:
    OperatorAdd(int mx_size, std::string name): 
        mx_size_(mx_size), 
        out_(new LSTensor<T>(mx_size)), 
        name_(name) {}
        
    void Forward(int size) {
        T* out_vec = out_->tensor();
        T* inp_A_vec = inp_A->tensor();
        T* inp_B_vec = inp_B->tensor();
        for (int i = 0; i < size; i ++) {
            out_vec[i] = inp_A_vec[i] + inp_B_vec[i];
        }
    }

    // void Backward(int size) {
    //     T* grad_inp_tensor = grad_inp_->tensor();
    //     T* grad_out_tensor = grad_out_->tensor();
    //     for(int i = 0 ; i < size; i ++) {
    //         grad_inp_tensor[i] = grad_out_tensor[i];
    //     }
    // }

    void init_forward(LSTensorPtr<T> tensor_a, LSTensorPtr<T> tensor_b) {
        LSTensorUsage::update_operator_idx();
        inp_A = tensor_a;
        inp_B = tensor_b;
        inp_A->update_life_idx();
        inp_B->update_life_idx();
        out_->update_life_idx();
    }

    // void init_backward(LSTensorPtr<T> tensor_) {
    //     LSTensorUsage::update_operator_idx();
    //     grad_out_ = tensor_;
    //     grad_out_->update_life_idx();
    //     grad_inp_->update_life_idx();
    // }

    void set_out(LSTensorPtr<T> tensor_) { 
        out_ = tensor_; 
    }
    LSTensorPtr<T> out() { return out_; }


    // void set_grad_out(LSTensorPtr<T> tensor_) { grad_out = tensor_; }
    // LSTensorPtr grad_out() { return grad_out; }
    // void set_grad_inp(LSTensorPtr<T> tensor_) { grad_inp = tensor_; }
    // LSTensorPtr grad_inp() { return grad_inp; }
    
};