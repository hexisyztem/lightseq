#include "transformer_encoder_layer.h"

#include "context.h"
#include "kernels.h"
#include "tensor.h"

template <typename T>
TransformerEncoderLayer<T>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, RuntimeStatus rs)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _max_batch_dim(max_batch_tokens * hidden_size),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _training(true),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _activation_fn(activation_fn),
      _qkv_linear(
          typename FeedForward<T>::Config(3 * hidden_size, hidden_size)),
      _attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
               _max_batch_tokens),
      _ffn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
              _max_batch_tokens),
      _ff1(typename FeedForward<T>::Config(_intermediate_size, hidden_size)),
      _ff2(typename FeedForward<T>::Config(hidden_size, _intermediate_size)),
      _softmax(typename Softmax<T>::Config(num_heads, mask_future_tokens)),
      _attn_prob_dropout(typename Dropout<T>::Config(attn_prob_dropout_ratio),
                         _max_batch_tokens * _heads * _max_seq_len),
      _attn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                    _max_batch_tokens * _hidden_size),
      _ffn_activation_dropout(
          typename Dropout<T>::Config(activation_dropout_ratio),
          _max_batch_tokens * _intermediate_size),
      _ffn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                   _max_batch_tokens * _hidden_size),
      _attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)),
      
      // fw-io
      _input_tensor(hidden_size * _max_batch_tokens, FixedMemory),
      _output_tensor(hidden_size * _max_batch_tokens, FixedMemory),
      // fw-attn
      _gemmQKV_inp_ptr(_max_batch_dim, SharedMemory),
      qkv_linear_output(3 * _max_batch_dim, SharedMemory),
      transform_qkv(3 * _max_batch_dim, SharedMemory),
      q_tf_ptr(transform_qkv.tensor(), _max_batch_dim, transform_qkv.unique_id()),
      k_tf_ptr(transform_qkv.tensor() + _batch_dim, _batch_dim, transform_qkv.unique_id()),
      v_tf_ptr(transform_qkv.tensor() + 2 * _batch_dim, _batch_dim, transform_qkv.unique_id()),
      _soft_out_ptr(_max_batch_tokens * _heads * _max_seq_len, SharedMemory),
      _ctx_bufB_ptr(_max_batch_tokens * _heads * _max_seq_len, SharedMemory),
      context_score(_max_batch_tokens * hidden_size, SharedMemory), 
      _attn_o_inp_ptr(_max_batch_tokens * hidden_size, SharedMemory), 
      attn_out_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      // fw-linear
      _ff1_inp_ptr(_max_batch_tokens * hidden_size, SharedMemory), 
      _relu_inp_ptr(_max_batch_tokens * intermediate_size, SharedMemory),
      _ff2_inp_ptr(_max_batch_tokens * intermediate_size, SharedMemory),

      // bw-io
      grad_output_tensor(_max_batch_tokens * hidden_size, FixedMemory),
      grad_input_tensor(_max_batch_tokens * hidden_size, FixedMemory),
      bw_output_tensor(_max_batch_tokens * hidden_size, FixedMemory),

      bw_input_tensor(_max_batch_tokens * hidden_size, SharedMemory),
      // bw-linear
      grad_ff2_out_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      grad_ffn_residual_ptr(_max_batch_tokens * hidden_size, SharedMemory),   
      grad_ff1_out_ptr(_max_batch_tokens * intermediate_size, SharedMemory),
      grad_ff1_inp_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      grad_attn_out_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      // bw-attn
      grad_attn_residual_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      grad_attn_linear_out(_max_batch_tokens * hidden_size, SharedMemory),
      grad_input_buf_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      grad_attn_ctx_out(_max_batch_tokens * hidden_size, SharedMemory),
      grad_qkv_5d_ptr(_max_batch_tokens * hidden_size * 3, SharedMemory),
      grad_q_5d_ptr(grad_qkv_5d_ptr.tensor(), _max_batch_dim, grad_qkv_5d_ptr.unique_id()),
      grad_k_5d_ptr(grad_qkv_5d_ptr.tensor() + _batch_dim, _batch_dim, grad_qkv_5d_ptr.unique_id()),
      grad_v_5d_ptr(grad_qkv_5d_ptr.tensor() + 2 * _batch_dim, _batch_dim, grad_qkv_5d_ptr.unique_id()),
      grad_softmax_ptr(_max_batch_tokens * hidden_size, SharedMemory),
      grad_qkv_4d_ptr(_max_batch_tokens * hidden_size * 3, SharedMemory),
      grad_input_buf_ptr(_max_batch_dim, SharedMemory),
      
       {
  assert(_hidden_size % _heads == 0);

  // attn_fw 
  if (_pre_or_postLayerNorm) {
    FW_REGIST_LIFE_CYCLE(_attn_ln.Forward, {_gemmQKV_inp_ptr, _input_tensor});
    FW_REGIST_LIFE_CYCLE(_qkv_linear.Forward, {_gemmQKV_inp_ptr, qkv_linear_output});
  }
  else {
    FW_REGIST_LIFE_CYCLE(_qkv_linear.Forward, {_input_tensor, qkv_linear_output});
  }
  FW_REGIST_LIFE_CYCLE(launch_bias_add_transform_20314, {transform_qkv, qkv_linear_output});
  FW_REGIST_LIFE_CYCLE(_attn_scores.Forward, {_soft_out_ptr, q_tf_ptr, k_tf_ptr});
  FW_REGIST_LIFE_CYCLE(_softmax.Forward, {_soft_out_ptr});
  FW_REGIST_LIFE_CYCLE(_attn_prob_dropout.dropout, {_ctx_bufB_ptr, _soft_out_ptr});
  FW_REGIST_LIFE_CYCLE(_attn_context.Forward, {context_score, v_tf_ptr, _ctx_bufB_ptr});
  FW_REGIST_LIFE_CYCLE(launch_transform4d_0213, {_attn_o_inp_ptr, context_score});
  FW_REGIST_LIFE_CYCLE(_attn_out_linear.Forward, {_attn_o_inp_ptr, attn_out_ptr});
  FW_REGIST_LIFE_CYCLE(_attn_dropout.bias_dropout_residual, {attn_out_ptr, attn_out_ptr, _input_tensor});
  if (!_pre_or_postLayerNorm) {
    FW_REGIST_LIFE_CYCLE(_attn_ln.Forward, {attn_out_ptr, attn_out_ptr});
  }

  // ffn_fw 
  if (_pre_or_postLayerNorm) { 
    FW_REGIST_LIFE_CYCLE(_ffn_ln, {_ff1_inp, _attn_out});
    FW_REGIST_LIFE_CYCLE(_ff1, {_ff1_inp, _relu_inp});
  }
  else {
    FW_REGIST_LIFE_CYCLE(_ff1, {_attn_out, _relu_inp});
  }
  FW_REGIST_LIFE_CYCLE(_ffn_activation_dropout, {_ff2_inp, _relu_inp});
  FW_REGIST_LIFE_CYCLE(_ff2, {_ff2_inp, _output_tensor});
  FW_REGIST_LIFE_CYCLE(_ffn_dropout, {_output_tensor, _output_tensor, _attn_out});
  if (!_pre_or_postLayerNorm) {
    FW_REGIST_LIFE_CYCLE(_ffn_ln, {_output_tensor, _output_tensor});
  }
  LSTensorUsage::update_operator_idx();

  if (rs == Training) {
    if(_pre_or_postLayerNorm) {
      BW_REGIST_LIFE_CYCLE(_ffn_dropout.d_bias_dropout_residual, {grad_ff2_out_ptr, grad_output_tensor}, {});
    }
    else {
      BW_REGIST_LIFE_CYCLE(_ffn_ln.Backward, {grad_ffn_residual_ptr, grad_output_tensor, bw_output_tensor}, {});
      BW_REGIST_LIFE_CYCLE(_ffn_dropout.d_bias_dropout_residual, {grad_ff2_out_ptr, grad_ffn_residual_ptr}, {});
    }
    BW_REGIST_LIFE_CYCLE(_ff2.Backward, {grad_ff2_out_ptr, grad_ff1_out_ptr}, {_ff2_inp_ptr});
    BW_REGIST_LIFE_CYCLE(_ffn_activation_dropout.d_bias_act_dropout, {grad_ff1_out_ptr}, {_relu_inp_ptr});
    BW_REGIST_LIFE_CYCLE();
  }



  LSTensor<T>::update_memory_manager();
}

template <typename T>
TransformerEncoderLayer<T>::~TransformerEncoderLayer() {}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_fw(LSTensor<T> attn_input_ptr,
                                               const T *input_mask_ptr,
                                               LSTensor<T> attn_out_ptr) {

  if (_pre_or_postLayerNorm) {
    _attn_ln.Forward(_gemmQKV_inp_ptr.tensor(), attn_input_ptr.tensor(), _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
    _qkv_linear.Forward(_batch_tokens, _gemmQKV_inp_ptr.tensor(), _attn_qkvw_ptr, qkv_linear_output.tensor(),
                      _cublasHandle);
  }
  else {
    _qkv_linear.Forward(_batch_tokens, attn_input_ptr.tensor(), _attn_qkvw_ptr, qkv_linear_output.tensor(),
                      _cublasHandle);
  }

  launch_bias_add_transform_20314<T>(transform_qkv.tensor(), qkv_linear_output.tensor(), _attn_qkvb_ptr,
                                     _batch_size, _seq_len, 3, _heads,
                                     _hidden_size / _heads, _stream);

  LSTensor<T> q_tf_ptr = LSTensor<T>(transform_qkv.tensor(), _batch_dim, transform_qkv.unique_id());
  LSTensor<T> k_tf_ptr = LSTensor<T>(transform_qkv.tensor() + _batch_dim, _batch_dim, transform_qkv.unique_id());
  LSTensor<T> v_tf_ptr = LSTensor<T>(transform_qkv.tensor() + 2 * _batch_dim, _batch_dim, transform_qkv.unique_id());

  // attention scores, q*k
  _attn_scores.Forward(_batch_heads, _soft_out_ptr.tensor(), k_tf_ptr.tensor(), q_tf_ptr.tensor(),
                       _cublasHandle);

  // Softmax + Mask
  _softmax.Forward(_soft_out_ptr.tensor(), input_mask_ptr, _batch_size, _seq_len,
                   _seq_len, _stream);

  // attn prob dropout.
  _attn_prob_dropout.dropout(_ctx_bufB_ptr.tensor(), _soft_out_ptr.tensor(),
                             _batch_heads * _seq_len * _seq_len, _stream);

  // attention context, score * v
  _attn_context.Forward(_batch_heads, _context_score.tensor(), v_tf_ptr.tensor(), _ctx_bufB_ptr.tensor(),
                        _cublasHandle);

  // [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T>(_attn_o_inp_ptr.tensor(), context_score.tensor(), _batch_size, _seq_len,
                             _hidden_size, _heads, 1, _stream);

  _attn_out_linear.Forward(_batch_tokens, _attn_o_inp_ptr.tensor(), _attn_ow_ptr,
                           attn_out_ptr.tensor(), _cublasHandle);

  _attn_dropout.bias_dropout_residual(attn_out_ptr.tensor(), attn_out_ptr.tensor(), attn_input_ptr.tensor(),
                                      _attn_ob_ptr, _batch_tokens, _hidden_size,
                                      _stream);
  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _attn_ln.Forward(attn_out_ptr.tensor(), attn_out_ptr.tensor(), _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_fw(LSTensor<T> inp_ptr, LSTensor<T> out_ptr) {
  // save _ff1_inp_ptr, _relu_inp_ptr, _ff2_inp_ptr for backward
  if (_pre_or_postLayerNorm) { // what if _pre_or_postLayerNorm is false
    _ffn_ln.Forward(_ff1_inp_ptr.tensor(), inp_ptr.tensor(), _ffn_nw_ptr, _ffn_nb_ptr,
                    _batch_tokens, _stream);
    _ff1.Forward(_batch_tokens, _ff1_inp_ptr.tensor(), _inter_w_ptr, _relu_inp_ptr.tensor(),
               _cublasHandle);
  }
  else {
    _ff1.Forward(_batch_tokens, inp_ptr.tensor(), _inter_w_ptr, _relu_inp_ptr.tensor(),
               _cublasHandle);
  }

  _ffn_activation_dropout.bias_act_dropout(
      _ff2_inp_ptr.tensor(), _relu_inp_ptr.tensor(), _inter_b_ptr, _batch_tokens,
      _intermediate_size, _activation_fn, _stream);

  _ff2.Forward(_batch_tokens, _ff2_inp_ptr.tensor(), _output_w_ptr, out_ptr.tensor(),
               _cublasHandle);

  _ffn_dropout.bias_dropout_residual(out_ptr.tensor(), out_ptr.tensor(), inp_ptr.tensor(), _output_b_ptr,
                                     _batch_tokens, _hidden_size, _stream);

  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _ffn_ln.Forward(out_ptr.tensor(), out_ptr.tensor(), _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens,
                    _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::Forward(const T *input_ptr,
                                         const T *input_mask_ptr, T *out_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();

  _input_tensor.set_tensor(input_ptr);
  _output_tensor.set_tensor(output_ptr);
  
  attn_layer_fw(input_tensor_, input_mask_ptr, attn_out_ptr);

  ffn_layer_fw(attn_out_ptr, _output_tensor);
}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_bw(LSTensor<T> input_ptr,
                                               const T *input_mask_ptr,
                                               LSTensor<T> grad_output_ptr,
                                               LSTensor<T> grad_input_ptr) {
  cudaStream_t streams[2] = {_stream, _stream};

  if (_pre_or_postLayerNorm) {
    _attn_dropout.d_bias_dropout_residual(grad_attn_linear_out.tensor(), _grad_attn_ob_ptr,
                                          grad_output_ptr.tensor(), _batch_tokens,
                                          _hidden_size, _stream);
  } else {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_attn_residual_ptr.tensor(),
                      grad_output_ptr.tensor().tensor(), nullptr, _ff1_inp_ptr.tensor(), _attn_nw_ptr,
                      _attn_nb_ptr, _batch_tokens, streams);
    _attn_dropout.d_bias_dropout_residual(grad_attn_linear_out.tensor(), _grad_attn_ob_ptr,
                                          grad_attn_residual_ptr.tensor(), _batch_tokens,
                                          _hidden_size, _stream);
  }

  // bw of output project
  _attn_out_linear.Backward(_batch_tokens, grad_attn_linear_out.tensor(), _attn_o_inp_ptr,
                            _attn_ow_ptr, _grad_attn_ow_ptr, _grad_attn_ob_ptr,
                            _cublasHandle, _stream, grad_input_buf_ptr.tensor(), nullptr,
                            false);
  launch_transform_0213<T>(grad_attn_ctx_out.tensor(), grad_input_buf_ptr.tensor(), _batch_size,
                           _seq_len, _hidden_size, _heads, _stream);

  // bw of score * v
  _attn_context.Backward(_batch_heads, grad_attn_ctx_out.tensor(), v_tf_ptr.tensor(), _ctx_bufB_ptr.tensor(),
                         _cublasHandle, grad_v_5d_ptr.tensor(),
                         grad_softmax_ptr.tensor());

  _attn_prob_dropout.d_dropout(grad_softmax_ptr.tensor(),
                               _batch_heads * _seq_len * _seq_len, _stream);

  _softmax.Backward(grad_softmax_ptr.tensor(), _soft_out_ptr.tensor(), _batch_size, _seq_len,
                    _seq_len, _stream);

  // bw of q * k
  _attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_tf_ptr.tensor(), q_tf_ptr.tensor(),
                        _cublasHandle, grad_k_5d_ptr.tensor(),
                        grad_q_5d_ptr.tensor());

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_qkv_4d_ptr.tensor(), grad_qkv_5d_ptr.tensor(), _batch_size,
                             _seq_len, _hidden_size, _heads, 3, _stream);

  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  

  if (_pre_or_postLayerNorm) {

    _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr.tensor(), _gemmQKV_inp_ptr.tensor(),
                       _attn_qkvw_ptr, _grad_attn_qkvw_ptr, _grad_attn_qkvb_ptr,
                       _cublasHandle, _stream, grad_input_buf_ptr.tensor());
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr.tensor(),
                      grad_input_buf_ptr.tensor(), grad_output_ptr, _gemmQKV_inp_ptr,
                      _attn_nw_ptr, _attn_nb_ptr, _batch_tokens, streams);
  } else {
    _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr.tensor(), input_ptr.tensor(),
                       _attn_qkvw_ptr, _grad_attn_qkvw_ptr, _grad_attn_qkvb_ptr,
                       _cublasHandle, _stream, grad_input_buf_ptr.tensor());
    // FIXME later
    launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr.tensor(), grad_attn_residual_ptr,
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}


template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_bw(LSTensor<T> grad_output_ptr,
                                              LSTensor<T> output_ptr,
                                              LSTensor<T> grad_inp_ptr) {
  cudaStream_t streams[2] = {_stream, _stream};

  if (_pre_or_postLayerNorm) {
    _ffn_dropout.d_bias_dropout_residual(grad_ff2_out_ptr.tensor(), _grad_output_b_ptr,
                                         grad_output_ptr.tensor(), _batch_tokens,
                                         _hidden_size, _stream);
  } else {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_ffn_residual_ptr.tensor(),
                     grad_output_ptr.tensor(), nullptr, output_ptr.tensor(), _ffn_nw_ptr,
                     _ffn_nb_ptr, _batch_tokens, streams);
    _ffn_dropout.d_bias_dropout_residual(grad_ff2_out_ptr.tensor(), _grad_output_b_ptr,
                                         grad_ffn_residual_ptr.tensor(), _batch_tokens,
                                         _hidden_size, _stream);
  }

  _ff2.Backward(_batch_tokens, grad_ff2_out_ptr.tensor(), _ff2_inp_ptr.tensor(), _output_w_ptr,
                _grad_output_w_ptr, _grad_output_b_ptr, _cublasHandle, _stream,
                grad_ff1_out_ptr.tensor(), nullptr, false);

  _ffn_activation_dropout.d_bias_act_dropout(
      grad_ff1_out_ptr.tensor(), _grad_inter_b_ptr, _relu_inp_ptr.tensor(), _inter_b_ptr,
      _batch_tokens, _intermediate_size, _activation_fn, _stream);

  _ff1.Backward(_batch_tokens, grad_ff1_out_ptr.tensor(), _ff1_inp_ptr.tensor(), _inter_w_ptr,
                _grad_inter_w_ptr, _grad_inter_b_ptr, _cublasHandle, _stream,
                grad_ff1_inp_ptr.tensor(), nullptr, false);

  /* ln signature:
  grad_gamma_grad, grad_betta, grad_inp,
  grad_out, grad_residual, output, gamma, betta,
  */
  const T *add_res_ptr = _ff1_inp_ptr;
  if (_pre_or_postLayerNorm) {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_inp_ptr.tensor(),
                     grad_ff1_inp_ptr.tensor(), grad_output_ptr.tensor(), _ff1_inp_ptr.tensor(),
                     _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens, streams);
  } else {
    launch_fused_add2<T>(grad_inp_ptr.tensor(), grad_ff1_inp_ptr.tensor(), grad_ffn_residual_ptr.tensor(),
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::Backward(const T *grad_output_ptr,
                                          const T *input_ptr,
                                          const T *output_ptr,
                                          const T *input_mask_ptr,
                                          T *grad_input_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *grad_ffn_inp_ptr = _shared_mem_ptr;
  T *buffer = grad_ffn_inp_ptr + _batch_dim;


  grad_output_tensor.set_tensor(grad_output_ptr);
  bw_input_tensor.set_tensor(input_ptr);
  bw_output_tensor.set_tensor(output_ptr);
  grad_input_tensor.set_tensor(grad_input_ptr);
  
  ffn_layer_bw(grad_output_tensor, bw_output_tensor, grad_attn_out_ptr);

  attn_layer_bw(bw_input_tensor, input_mask_ptr, grad_attn_out_ptr, grad_input_tensor);
}

template <typename T>
void TransformerEncoderLayer<T>::SetTrainingMode(bool training) {
  // Dropout will be skipped when not in training model.
  _attn_prob_dropout.SetTrainingMode(training);
  _attn_dropout.SetTrainingMode(training);
  _ffn_activation_dropout.SetTrainingMode(training);
  _ffn_dropout.SetTrainingMode(training);
}

template <typename T>
T *TransformerEncoderLayer<T>::_shared_mem_ptr = nullptr;

template class TransformerEncoderLayer<float>;
template class TransformerEncoderLayer<__half>;
