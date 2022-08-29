#include "layer_normalize.h"

template <typename T>
Layer_Normalize<T>::Layer_Normalize(Config config, size_t max_rows)
    : config_(config), vars_(nullptr), means_(nullptr) {
  vars_ = cuda_malloc<T>(max_rows);
  if (config_.use_mean) {
    means_ = cuda_malloc<T>(max_rows);
  }
}

template <typename T>
Layer_Normalize<T>::~Layer_Normalize() {
  cuda_free(vars_);
  cuda_free(means_);
}

template <typename T>
void Layer_Normalize<T>::Forward(T *ln_res, const T *inp, const T *gamma,
                                 const T *betta, int batch_size,
                                 cudaStream_t stream) {
  launch_layer_norm(ln_res, vars_, means_, inp, gamma, betta, batch_size,
                    config_.hidden_dim, stream);
}

template <typename T>
void Layer_Normalize<T>::Backward(T *gamma_grad, T *betta_grad, T *inp_grad,
                                  const T *out_grad, const T *residual_grad,
                                  const T *inp_or_out, const T *gamma,
                                  const T *betta, int batch_size,
                                  cudaStream_t stream[2]) {
  launch_ln_bw(gamma_grad, betta_grad, inp_grad, out_grad, residual_grad,
               inp_or_out, gamma, betta, vars_, means_, batch_size,
               config_.hidden_dim, stream);
}

template <typename T>
inline bool Layer_Normalize<T>::use_mean() const {
  return config_.use_mean;
}

template class Layer_Normalize<float>;
template class Layer_Normalize<__half>;
