// #pragma once

#include <string>
#include <memory>
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

template<class T>
class MemoryManager {
private:
    static T* buffer_;
    std::map<int, LSTensorUsage> tensor_usages_;
    size_t buffer_size_;
    std::map<int, T*> tensor_ptr;
public:
    MemoryManager(){}
    ~MemoryManager() { }

    T* get_memory(int unique_id) {
        return tensor_ptr.find(unique_id)->second;    
    }

    void set_first_idx(int unique_id, size_t size) {
        std::map<int, LSTensorUsage>::iterator iter = tensor_usages_.find(unique_id);
        if (iter != tensor_usages_.end()) {
            return ;
        }
        int first_idx = LSTensorUsage::operator_idx;
        int last_idx = first_idx + 1;
        tensor_usages_.emplace(unique_id, LSTensorUsage(first_idx, last_idx, size));
        return ;
    }

    
    void update_last_idx(int unique_id) {
        std::map<int, LSTensorUsage>::iterator iter = tensor_usages_.find(unique_id);
        if (iter == tensor_usages_.end()) {
            printf("error orrcured!");
            exit(0);
        }
        iter->second.last_idx = LSTensorUsage::operator_idx;
    }

    void remove_life_cycle(int unique_id) {
        tensor_usages_.erase(unique_id);
    }

    void update() {
        if (buffer_ != nullptr) {
            free(buffer_);
        }
        tensor_ptr.clear();
        size_t tmp_buffer_size_ = 0;
        for(auto iter: tensor_usages_) {
            tmp_buffer_size_ += iter.second.size;
        }
        buffer_ = (T*)malloc(tmp_buffer_size_ * sizeof(T));

        buffer_size_ = tmp_buffer_size_ * sizeof(T);
        T* tmp_pointer = buffer_;
        for(auto iter: tensor_usages_) {
            int unique_id = iter.first;
            tensor_ptr.emplace(unique_id, tmp_pointer);
            size_t size = iter.second.size;
            tmp_pointer = tmp_pointer + size;
            printf("inner update %zu\n", size);
        }
    }
    size_t buffer_size() { return buffer_size_;}
};

template<class T>
T* MemoryManager<T>::buffer_ = nullptr;



template <typename T>
class LSTensor {
private:
    LSMemoryType memory_type_;
    T* tensor_ = nullptr;
    int unique_id_;
    size_t tensor_size_;

    static int tensor_id_;
    static MemoryManager<T> memory_manager_;
public:
    LSTensor() = delete;
    LSTensor(int size, LSMemoryType mt = SharedMemory) { // intermediate tensor
        unique_id_ = tensor_id_ ++;
        tensor_size_ = size;
        memory_type_ = mt;
    }
    // subtensor for attention transform
    LSTensor(T* tensor_pointer, int size, uintptr_t father_id = 0) {
        tensor_ = tensor_pointer;
        if(father_id != 0){
            unique_id_ = tensor_id_;
            memory_type_ = FixedMemory;
        }
        else {
            unique_id_ = tensor_id_ ++;
            memory_type_ = SharedMemory;
        }
        tensor_size_ = size;
    }
    LSTensor(LSTensor& a) = default;
    ~LSTensor() {
        memory_manager_.remove_life_cycle(unique_id_);
    }

    void set_tensor(T* inp) { tensor_ = inp; }
    void set_tensor(const T* inp) { tensor_ = const_cast<T*>(inp); }

    T* tensor() {
        if (tensor_ == nullptr) {
            tensor_ = memory_manager_.get_memory(unique_id_);
        }
        return tensor_;
    }

    size_t size() { return tensor_size_; }

    int unique_id() { return unique_id_; }

    void update_life_idx(){ 
        if (memory_type_ == FixedMemory) {
            return ;
        }
        memory_manager_.set_first_idx(unique_id_, tensor_size_); 
        memory_manager_.update_last_idx(unique_id_);
    }

    static void update_memory_manager() {
        memory_manager_.update();
        printf("update_memory_manager, total buffer size: %zu\n", memory_manager_.buffer_size());
    }
};

template<typename T>
MemoryManager<T> LSTensor<T>::memory_manager_ = MemoryManager<T>();


template<typename T>
int LSTensor<T>::tensor_id_ = 0;

template<typename T>
using LSTensorPtr = std::shared_ptr<LSTensor<T>>;