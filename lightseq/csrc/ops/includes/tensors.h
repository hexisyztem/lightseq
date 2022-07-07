#pragma once

#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "cuda_util.h"
#include "kernels.h"
#include "memory"
#include <map>

enum LSMemoryType {
    FixedMemory,
    SharedMemory
};

enum RuntimeStatus {
    Inference,
    Training
};

class LSTensorUsage {
public:
    int first_idx, last_idx;
    size_t size;
    static int operator_idx;
    LSTensorUsage(int fidx, int lidx, size_t s): first_idx(fidx), last_idx(lidx), size(s) {}
    static void update_operator_idx() { operator_idx ++; }
};

int LSTensorUsage::operator_idx = -1;

class MemoryManager {
private:
    static void* buffer_;
    std::map<uintptr_t, LSTensorUsage> tensor_usages_;
    int buffer_size_;
    std::map<uintptr_t, void*> tensor_ptr;
public:
    MemoryManager(){}
    ~MemoryManager() { }

    void* get_memory(uintptr_t unique_id) {
        return tensor_ptr.find(unique_id)->second;    
    }

    void set_first_idx(int unique_id, size_t size) {
        std::map<uintptr_t, LSTensorUsage>::iterator iter = tensor_usages_.find(unique_id);
        if (iter != tensor_usages_.end()) {
            return ;
        }
        int first_idx = LSTensorUsage::operator_idx;
        int last_idx = first_idx + 1;
        tensor_usages_.emplace(unique_id, LSTensorUsage(first_idx, last_idx, size));
        return ;
    }

    
    void update_last_idx(uintptr_t unique_id) {
        std::map<uintptr_t, LSTensorUsage>::iterator iter = tensor_usages_.find(unique_id);
        if (iter == tensor_usages_.end()) {
            //catch error;
            printf("error orrcured!");
            exit(0);
        }
        iter->second.last_idx = LSTensorUsage::operator_idx;
    }

    void remove_life_cycle(uintptr_t unique_id) {
        tensor_usages_.erase(unique_id);
    }

    void update() {
        tensor_ptr.clear();
        size_t total_size = 0;
        for(auto iter: tensor_usages_) {
            total_size += iter->second.size;
        }
        buffer = (void*)malloc(total_size);
        void* tmp_pointer = buffer;
        for(auto iter: tensor_usages_) {
            int unique_id = iter->first;
            tensor_ptr.emplace(unique_id, tmp_pointer);
            size_t size = iter->second.size;
            tmp_pointer += size;
        }
    }
};

template <typename T>
class LSTensor {
private:
    LSMemoryType memory_type_;
    T* tensor_ = nullptr;
    int unique_id_;
    size_t tensor_size_;

    static MemoryManager memory_manager_;
    static int tensor_id_;

public:
    LSTensor() = delete;
    LSTensor(int size, LSMemoryType mt = SharedMemory) { // intermediate tensor
        unique_id_ = tensor_id_ ++;
        tensor_size_ = size * sizeof(T);
        memory_type_ = mt;
    }
    // subtensor for attention transform
    LSTensor(T* tensor_pointer, int s, uintptr_t father_id = 0) {
        tensor_ = tensor_pointer;
        if(father_id != 0){
            unique_id_ = tensor_id_;
            memory_type_ = FixedMemory;
        }
        else {
            unique_id_ = tensor_id_ ++;
            memory_type_ = SharedMemory;
        }
        tensor_size_ = s * sizeof(T);
    }
    LSTensor(LSTensor& a) = default;

    void set_tensor(T* inp) { tensor_ = inp; }
    void set_tensor(const T* inp) { tensor_ = const_cast<T*>(inp); }

    T* tensor() {
        if (tensor_ == nullptr) {
            tensor_ = reinterpret_cast<T*>(memory_manager_.get_memory(unique_id));
        }
        return tensor_;
    }

    uintptr_t unique_id() { return unique_id_; }

    void update_life_idx(){ 
        if (memory_type_ == FixedMemory) {
            return ;
        }
        memory_manager_.set_first_idx(unique_id_, tensor_size_); 
        memory_manager_.update_last_idx(unique_id_);
    }

    void trans_fixed_memory() {
        
        if (memory_type_ == FixedMemory) {
            return ;
        }
        memory_manager_.remove_life_cycle(unique_id_);
        tensor_ = (T*)malloc(tensor_size_ * sizeof(T));
    }

    static void update_memory_manager() {
        memory_manager_.update();
    }
};

template<typename T>
MemoryManager LSTensor<T>::memory_manager_ = MemoryManager();

template<typename T>
int LSTensor<T>::tensor_id_ = 0;


template<class T>
void regist_life_cycle(std::string oper_name, std::vector<LSTensor<T>> intermediate_tensor, std::vector<LSTensor<T>> checkpoint_tensor) {
    LSTensorUsage::update_operator_idx();

    for(auto tensor_: intermediate_tensor) {
        tensor_.update_life_idx();
    }

    for(auto tensor_: checkpoint_tensor) {
        tensor_.trans_fixed_memory();
    }
}

#define FW_REGIST_LIFE_CYCLE(func, tensor_) regist_life_cycle(#func, tensor_, {})

#define BW_REGIST_LIFE_CYCLE(func, intermediate, checkpoint) regist_life_cycle(#func, intermediate, checkpoint)
