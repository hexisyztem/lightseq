#include "manager.cpp"
#include "vector"

template<class T>
class Operator {
private:
    LSTensorPtr<T> inp_ = nullptr, out_ = nullptr;
    LSTensorPtr<T> grad_inp_ = nullptr;
    LSTensorPtr<T> grad_out_ = nullptr;
    int mx_size_;

public:
    Operator(int mx_size): mx_size_(mx_size), out_(new LSTensor<T>(mx_size)) {

    }
    void Forward(int val, int size) {
        T* inp_tensor = inp_->tensor();
        T* out_tensor = out_->tensor();
        for (int i = 0; i < size; i ++) {
            out_tensor[i] = inp_tensor[i] + val;
        }
    }

    void Backward(int size) {
        T* grad_inp_tensor = grad_inp_->tensor();
        T* grad_out_tensor = grad_out_->tensor();
        for(int i = 0 ; i < size; i ++) {
            grad_inp_tensor[i] = grad_out_tensor[i];
        }
    }

    void init_forward(LSTensorPtr<T> tensor_) {
        LSTensorUsage::update_operator_idx();
        inp_ = tensor_;
        inp_->update_life_idx();
        out_->update_life_idx();
    }

    void init_backward(LSTensorPtr<T> tensor_) {
        LSTensorUsage::update_operator_idx();
        grad_out_ = tensor_;
        grad_out_->update_life_idx();
        grad_inp_->update_life_idx();
    }

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
class Layer {
private:
    Operator<T> op_a;
    Operator<T> op_b;
    Operator<T> op_c;

    LSTensorPtr<T> inp;
    LSTensorPtr<T> out;

    int mx_size_;

public:
    Layer(int mx_size, RuntimeStatus rs):
        mx_size_(mx_size),
        inp(new LSTensor<T>(mx_size, FixedMemory)), 
        out(new LSTensor<T>(mx_size, FixedMemory)),
        op_a(mx_size),
        op_b(mx_size),
        op_c(mx_size) {
        init_forward();
        // init_backward();
    }

    void init_forward() {
        op_a.init_forward(inp);
        op_b.init_forward(op_a.out());
        op_c.init_forward(op_b.out());
        op_c.set_out(out);
    }

    void init_backward() {

    }

    void Forward(int size, const T* inp_ptr, T* out_ptr) {
        inp->set_tensor(inp_ptr);
        out->set_tensor(out_ptr);

        op_a.Forward(1, size);

        op_b.Forward(10, size);
        op_c.Forward(100, size);
    }

    void Backward() {

    }
};

// template<class T>
// static std::vector<Layer<T>> 

int main(){
    Layer<int> layer_a = Layer<int>(20, Inference);
    int* inp_ = (int*)malloc(10 * sizeof(int));
    int* out_ = (int*)malloc(10 * sizeof(int));
    for(int i = 0; i < 10; i ++) 
        inp_[i] = 1;
    LSTensor<int>::update_memory_manager();
    layer_a.Forward(10, inp_, out_);

    for(int i = 0; i < 10; i ++){
        printf("%d ", *(out_ + i));
    }
}