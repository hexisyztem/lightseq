## 使用说明 - 中文版

### Reference
- [triton-inference-server/backend](https://github.com/triton-inference-server/backend)
- [triton-inference-server/server](https://github.com/triton-inference-server/server)
- [triton-inference-server/client](https://github.com/triton-inference-server/client)
- [triton-inference-server/core](https://github.com/triton-inference-server/core)
- [triton-inference-server/common](https://github.com/triton-inference-server/common)

### 如何使用 triton-lightseq-backend

- 获取 tritonserver: [tritonserver Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md#install-triton-docker-image)
  > $ docker pull nvcr.io/nvidia/tritonserver:22.01-py3
- 将 libtriton_lightseq.so 和 libliblightseq.so 放入模型文件的对应位置
    > 将上述 libtriton_lightseq.so 放入 <path_to_model_repository>/<model_name> 中 \
    > 并将 libliblightseq.so 放入 <path_to_model_repository> 中
- 将模型文件放入对应文件
    > 以 bert_example 为例，将模型文件存放入 <path_to_model_repository>/<model_name>/<version_id> 中
- 模型配置文件参数详解
    > \${name}: 模型名称，该字段需要跟 <model_name> 文件名对齐 \
    > \${backend}: "lightseq"，该字段用于识别backend的动态链接库文件 libtriton_lightseq.so \
    > \${default_model_filename}: 模型文件名，用于识别加载模型参数 \
    > \${parameters:model_type:string_value}: 用于识别模型类别，用于对应加载 lightseq 模型结构 \
    > \${version_policy:specific:versions}: 用于读取需要加载的模型版本，对应 <version_id>
- docker 运行命令:
    > $ docker run --gpus=1 --rm -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/models" -p8000:8000 -p8001:8001 -p8002:8002 -v<path_to_model_repository>:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models \
    > 其中 $LD_LIBRARY_PATH 用于将 libliblightseq.so 添加进入动态链接库的搜索路径