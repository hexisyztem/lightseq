#include "base_class.h"
#include "layer.h"
#include "manager.h"
#include "operator.h"
#include "iostream"
#include "vector"




int main(){
    Layer<int> layer_a = Layer<int>(20, "layer_a");
    Layer<int> layer_b = Layer<int>(20, "layer_b");
    int* inp_ = (int*)malloc(10 * sizeof(int));
    
    int* out_1 = (int*)malloc(10 * sizeof(int)); // layer output

    int* out_2 = (int*)malloc(10 * sizeof(int));
    for(int i = 0; i < 10; i ++) 
        inp_[i] = 1;
    std::cout << "address1: " << inp_ << std::endl;
    layer_a.Forward(10, inp_, out_1);
    layer_b.Forward(10, out_1, out_2);

    for(int i = 0; i < 10; i ++){
        printf("%d ", *(out_2 + i));
    }

    int* grad_out_ptr = (int*)malloc(10 * sizeof(int));
    int* grad_inp_ptr = (int*)malloc(10 * sizeof(int));

    
}