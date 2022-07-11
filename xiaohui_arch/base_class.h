#pragma once
#include "manager.h"

template<class T>
class LayerBase {
protected:
    static std::vector<LayerBase<T>*> LayerVec;
    static bool is_constructed_;
    static RuntimeStatus rs_;

    std::string name_;

public:
    LayerBase(std::string name): name_(name) {}
    static void construct_life_cycle();

    virtual void init_forward() {}

    virtual void init_backward() {}
};

template<class T>
std::vector<LayerBase<T>*> LayerBase<T>::LayerVec = {};

template<class T>
bool LayerBase<T>::is_constructed_ = false;

template<class T>
RuntimeStatus LayerBase<T>::rs_ = Inference;

template<class T>
void LayerBase<T>::construct_life_cycle() {
    if (is_constructed_) {
        return ;
    }
    printf("Running construct_life_cycle\n");
    is_constructed_ = true;
    // std::vector<LayerBase<T>*>::iterator layer_pointer = LayerVec.begin();
    // while (layer_pointer != LayerVec.end()) {
    //     std::cout << "layer_pointer->name_ " << layer_pointer->name_ << std::endl;
    //     layer_pointer->init_forward();
    //     layer_pointer.next();
    // }
    for(int i = 0; i < LayerVec.size(); i ++) {
        auto& layer_pointer = LayerVec[i];
        std::cout << "layer_pointer->name_ " << layer_pointer->name_ << std::endl;
        layer_pointer->init_forward();
    }
    if (rs_ == Training) {
        for(int i = LayerVec.size() - 1; i >= 0; i --) {
            auto& layer_pointer = LayerVec[i];
            std::cout << "layer_pointer->name_ " << layer_pointer->name_ << std::endl;
            layer_pointer->init_backward();
        }
    }
    LSTensor<int>::update_memory_manager();
}