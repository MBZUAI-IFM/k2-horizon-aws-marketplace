# K2 Horizon 7B FP8 model details

## Model and training

This offering uses [IFM/K2-Horizon-7B-FP8](https://huggingface.co/IFM/K2-Horizon-7B-FP8), revision `6cf6eab2661c549b1ecbe508d5e243a4dc843317`. K2 Horizon 7B is a dense, decoder-only language model. The checkpoint stores quantized linear weights in FP8, except for the output head. The NVIDIA L4 deployment uses FP8 weights with BF16 activations and KV cache.

IFM reports approximately 20 trillion pretraining tokens per Horizon model, drawn from web, code, mathematics, science, multilingual and domain-specific material, including approximately 10 trillion synthetic tokens. Further training develops long-context, reasoning and agentic capabilities. Data and recipe availability varies with redistribution rights; datasets retain their own licenses. See the [training overview](https://ifm.ai/blog/k2/) and [model training-stage inventory](https://huggingface.co/IFM/K2-Horizon-7B#training-overview).

## Upstream evaluation

The FP8 model card reports the following K2 Horizon 7B reference scores:

| Benchmark | Reported score |
| --- | ---: |
| HMMT February 2026 | 73.3% |
| SWE-bench Verified | 70.6% |
| Humanity's Last Exam | 18.6% |

These are upstream model-card results. They have not been reproduced on this AWS Marketplace deployment, and the table does not identify a separate FP8-on-L4 evaluation. Harness, context and output budgets affect results. Consult the [source evaluation table and settings](https://huggingface.co/IFM/K2-Horizon-7B-FP8#benchmark-results) before comparing scores.

## Observed deployment latency

A single streaming arithmetic request on the 32K configuration returned the correct answer, `1081`, on `ml.g6.2xlarge` with one NVIDIA L4:

| Measurement | Observation |
| --- | ---: |
| Formatted input tokens | 23 |
| Completion tokens, including reasoning | 102 |
| Client-observed request duration | 3.968 seconds |
| Concurrent sequences | 1 |

The request asked “Calculate 23 times 47. Return only the integer result.” It used high reasoning effort, temperature 1.0, top_p 0.95 and a 16,384-token generation allowance. The endpoint was already serving; startup time is excluded. The duration includes client/network overhead. This observation is a functional sample, not a latency percentile, throughput benchmark or service-level commitment. The [sample input and measurement](https://github.com/MBZUAI-IFM/k2-horizon-aws-marketplace/tree/main/samples/k2-horizon-7b-fp8) are provided with the deployment examples.

## Limits and operating behavior

The notebook configures the 32K profile explicitly. For a manual real-time deployment, pass these values in the model container environment; selecting `ml.g6.2xlarge` by itself does not configure the context budget:

```json
{
  "K2_SERVED_MODEL_NAME": "k2-horizon-7b",
  "K2_MAX_MODEL_LEN": "32768",
  "K2_MAX_NUM_SEQS": "1",
  "K2_TENSOR_PARALLEL_SIZE": "1",
  "K2_MAX_NUM_BATCHED_TOKENS": "512",
  "K2_ENFORCE_EAGER": "1",
  "K2_GPU_MEMORY_UTILIZATION": "0.90",
  "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
  "HF_HUB_OFFLINE": "1",
  "TRANSFORMERS_OFFLINE": "1",
  "HF_DATASETS_OFFLINE": "1"
}
```

Use the deployment configurations listed in this repository and in the selected Marketplace version. The upstream model's native context window does not establish the supported context of a particular AWS deployment. Input, history, tool definitions, generated reasoning and final output share the configured context budget.

The example's 16,384-token generation allowance is smaller than the upstream evaluation recommendation. Complex reasoning may exhaust it; inspect `finish_reason` and evaluate your intended workloads. The separate reasoning-token usage field is unreliable in this runtime.

The model accepts text. Generated facts, code and tool arguments require application validation. Tool execution, document retrieval and conversation storage are application responsibilities. Deployment capacity and quotas depend on the AWS Region and account.

Software is supplied under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0); AWS infrastructure charges apply. Support: [support@ifm.ai](mailto:support@ifm.ai).
