#include "bert.h"

namespace lightseq {
namespace cuda {

Bert::Bert(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"encoder_output"}),
      _max_batch_size(max_batch_size) {
  /* --- step.1 initial context --- */

  context_ptr.reset(new Context());
  Context::set_thread_context(context_ptr);

  /* --- step.2 load model weights into GPU memory --- */

  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }

  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */

  inp_tokens = new Variable("inp_tokens");
  token_emb = new Variable("token_emb", (char *)tw_.get_src_emb_wei()[0]);
  pos_emb = new Variable("pos_emb", (char *)tw_.get_src_emb_wei()[1]);
  pad_mask_ptr = cuda_malloc<int>(_max_batch_size * tw_._max_step);
  pad_mask = new Variable("pad_mask", (char *)pad_mask_ptr);
  lang_emb = new Variable("lang_emb", (char *)tw_.get_src_emb_wei()[4]);
  lang_id = new Variable("lang_id", nullptr);

  /* --- step.4 initial layer weight --- */

  int enc_wei_offset = 0;
  int src_emb_offset = 0;
  for (int i = 0; i < tw_._n_enc_layer; i++) {
    TransformerEncoderLayerWeightPtr enc_lyr_wt_(
        new TransformerEncoderLayerWeight(tw_._hidden_size, tw_._inner_size));
    enc_lyr_wt_->load_params(tw_.get_enc_wei(), enc_wei_offset);
    enc_layer_wts.push_back(enc_lyr_wt_);
  }
  LyrNormalizeLayerWeightPtr lyr_norm_wt(
      new LyrNormalizeLayerWeight(tw_._hidden_size));
  lyr_norm_wt->load_params(tw_.get_src_emb_wei(), 2);

  /* --- step.5 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  // initial launch_enc_emb_op
  launch_enc_emb_op = new LaunchEncEmbOp<OpType_>(
      max_batch_tokens, tw_._padding_id, tw_._hidden_size, tw_._multilg_type);

  float attn_prob_dropout_ratio = 0.0;
  float activation_dropout_ratio = 0.0;
  float hidden_dropout_ratio = 0.0;

  // initial transformer encoder layers
  for (int idx = 0; idx < tw_._n_enc_layer; idx++) {
    TransformerEncoderLayerPtr<OpType_, OpType_> enc_layer_(
        new TransformerEncoderLayer<OpType_, OpType_>(
            enc_layer_wts[idx], idx, max_batch_tokens, tw_._max_step,
            tw_._hidden_size, tw_._head_num, tw_._inner_size,
            attn_prob_dropout_ratio, activation_dropout_ratio,
            hidden_dropout_ratio, true, tw_._use_gelu ? "gelu" : "relu", false,
            tw_._is_post_ln));
    enc_layer_vec.push_back(enc_layer_);
  }

  lyr_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      lyr_norm_wt, max_batch_tokens, tw_._hidden_size));

  /* --- step.6 construct network --- */
  Variable *enc_emb = (*launch_enc_emb_op)(inp_tokens, token_emb, pos_emb,
                                           pad_mask, lang_emb, lang_id);
  for (auto iter : enc_layer_vec) {
    enc_emb = (*iter)(enc_emb, pad_mask);
    std::cout << "enc_emb address: " << enc_emb << std::endl;
  }
  bert_out = (*lyr_norm_layer)(enc_emb);
}

Bert::~Bert() { cuda_free(pad_mask_ptr); }

void Bert::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  // before forward
  launch_enc_emb_op->before_forward(batch_size, seq_len);
  for (auto iter : enc_layer_vec) {
    iter->before_forward(batch_size, seq_len);
  }
  lyr_norm_layer->before_forward(batch_size * seq_len);

  launch_enc_emb_op->recursive_forward();
  for (auto iter : enc_layer_vec) {
    iter->forward();
  }
  lyr_norm_layer->forward();

  set_output_shape(0, {batch_size, seq_len, tw_._hidden_size});
}

void Bert::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      inp_tokens->set_value((char *)input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Bert::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      bert_out->set_value((char *)output_ptr);
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Bert::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(bert_out->value());
    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Bert::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> Bert::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step, tw_._hidden_size};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Bert::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Bert::get_output_dtype(int index) {
  switch (index) {
    case 0:
#ifdef FP16_MODE
      return DataType::kFloat16;
#else
      return DataType::kFloat32;
#endif

      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
