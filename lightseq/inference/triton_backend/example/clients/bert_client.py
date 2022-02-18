#!/usr/bin/env python
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
np.set_printoptions(threshold=np.inf)

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    FLAGS = parser.parse_args()

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # Send 2 requests to the batching model. Because these are sent
    # asynchronously and Triton's dynamic batcher is configured to
    # delay up to 5 seconds when forming a batch for this model, we
    # expect these 2 requests to be batched within Triton and sent to
    # the backend as a single batch.
    #
    # The recommended backend can handle any model with 1 input and 1
    # output as long as the input and output datatype and shape are
    # the same. The batching model uses datatype FP32 and shape
    # [ 4, 4 ].
    print('\n=========')
    async_requests = []

    # bert_pad_input = [[ 101, 4931, 1010, 2129, 2024, 2017, 102, 0 ]]
    # gpt_pad_input = [[3666, 1438, 318, 402, 11571]]
    transformer_pad_input = [[0,     100, 657, 14,    1816, 6, 53, 50264, 473, 45,  50264, 162,  4, 2],
                             [0,     100, 657, 14,    1816, 6, 53, 50264, 473, 45,  50264, 162,  4, 2]]
    pad_input = transformer_pad_input
    input0_data = np.array(pad_input, dtype=np.int32)
    print('Sending request to lightseq_bert_base_uncased model: input = {}'.format(input0_data))
    inputs = [ httpclient.InferInput('source_ids', [2, 14], "INT32") ]
    inputs[0].set_data_from_numpy(input0_data)
    async_requests.append(triton_client.async_infer('lightseq_test_models', inputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference
        # request. This call will block till the server responds.
        result = async_request.get_result()
        print('Response: {}'.format(result.get_response()))
        print('target_ids = {}'.format(result.as_numpy('target_ids')))
        print('target_scores = {}'.format(result.as_numpy('target_scores')))
