#include "launch_concat3_dim1.h"

namespace lightseq {

template <typename T1, typename T2>
Variable LaunchConcat3Dim1<T1, T2>::operator()(Variable* inp,
                                               Variable* chache) {
  Variable* new_cache = new Variable(this->_name + "/new_cache");
  this->set_parents({inp, cache});
  this->set_children({new_cache});
  return new_cache;
}

template <typename T1, typename T2>
void LaunchConcat3Dim1<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* cache_ptr = (T1*)parent(1)->value();
  T1* real_val = (T1*)child(0)->value();
  CHECK_GPU_ERROR(cudaMemcpyAsync((void*)real_val, (void*)inp_ptr,
                                  _batchs * _hidden_size * seq_len * sizeof(T1),
                                  cudaMemcpyDefault, _stream));
  if (_predict) {
    launch_concat3_dim1(real_val, inp_ptr, cache_ptr, _batchs * _heads,
                        _hidden_size / _heads, _steps, 1, _stream);
  }

#ifdef DEBUG
  if (_context_ptr->built()) {
    cudaStreamSynchronize(_stream);
    printf("%s forward is_predict: %d\n", name().c_str(), _predict);
    print_vec(inp_ptr, "inp_ptr", 10);
    print_vec(cache_ptr, "cache_ptr", 10);
    print_vec(real_val, "real_val", 10);
    printf("\n");
  }
#endif
}

template <typename T1, typename T2>
void LaunchConcat3Dim1<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* val_grad = (T1*)child(0)->grad();
  CHECK_GPU_ERROR(cudaMemcpyAsync((void*)inp_grad, (void*)val_grad,
                                  _batchs * _hidden_size * seq_len * sizeof(T2),
                                  cudaMemcpyDefault, _stream));
}

template class LaunchConcat3Dim1<float, float>;
template class LaunchConcat3Dim1<__half, __half>;

}  // namespace lightseq
