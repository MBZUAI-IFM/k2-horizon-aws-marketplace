# K2 Horizon on AWS Marketplace

Deploy K2 Horizon models through AWS Marketplace and Amazon SageMaker AI.

These examples deploy models in your AWS account and demonstrate chat, streaming, conversation history, tool calling and batch inference. You control the deployment and its lifecycle.

**Availability:** Marketplace publication is in progress. Run the examples once the product and its regional model package are available to your account.

## Examples

| Model | Notebook | Python helpers |
| --- | --- | --- |
| K2 Horizon 7B FP8 | [SageMaker notebook](notebooks/k2-horizon-7b-fp8.ipynb) | [Deployment and inference helpers](examples/k2-horizon-fp8-sagemaker.py) |

The first example uses the [K2 Horizon 7B FP8 checkpoint](https://huggingface.co/IFM/K2-Horizon-7B-FP8). Its configurations are specific to that checkpoint; future model examples may use different precision, hardware and settings.

| Configuration | Total context | SageMaker instance | GPUs |
| --- | ---: | --- | --- |
| `32k` | 32,768 tokens | `ml.g6.2xlarge` | 1 NVIDIA L4 |
| `64k` | 65,536 tokens | `ml.g6.12xlarge` | 4 NVIDIA L4 |

Use only configurations supported by the selected Marketplace version. The total context includes input, conversation history, tool definitions, generated reasoning and the final answer. This example uses one active sequence, FP8 weights and BF16 activations and KV cache.

## Get started

1. Subscribe to the K2 Horizon product in AWS Marketplace and accept its terms when available. Copy the model package ARN for your deployment Region from the product’s deployment instructions.
2. Open the notebook in an Amazon SageMaker AI notebook environment with Python 3. Install the dependencies in the notebook’s kernel:

   ```python
   %pip install boto3 "sagemaker>=2.254,<3"
   ```

   Alternatively, clone this repository and install `requirements.txt` in your Python environment.

3. Set `PACKAGE_ARN` in the setup cell, or provide the `K2_MODEL_PACKAGE_ARN` environment variable. Select `DEPLOYMENT_PROFILE`; the default is `32k`.
4. Run the cells in order. The notebook deploys an endpoint, runs the inference examples and cleans up the resources it creates. The batch section runs a separate temporary job; set `RUN_BATCH=False` to skip it.

The notebook uses the standard AWS credential chain and detects the SageMaker execution role. Outside SageMaker, configure AWS credentials and set `SAGEMAKER_EXECUTION_ROLE_ARN`. Never put access keys in the notebook.

Your AWS identity needs permission to create, inspect, invoke and delete SageMaker resources, and to pass the execution role. The execution role needs access to the model package and the example S3 input/output prefix. Endpoint and Batch Transform quotas are separate, and deployment depends on regional instance availability.

## Costs and cleanup

The model has no software fee. AWS compute, storage and other infrastructure charges apply.

Keep the notebook running until cleanup confirms deletion. If execution is interrupted, retain the printed resource names and rerun `cleanup(session, resources)` or `cleanup_batch(session, batch_resources)`. Verify that the endpoint is deleted or the batch job is terminal before leaving. A local timeout does not stop AWS billing.

Setting `EXISTING_ENDPOINT` invokes an endpoint you own without deleting it. The batch example preserves your S3 bucket; object versions may remain if bucket versioning is enabled.

## Model and support

- [Institute of Foundation Models](https://ifm.ai)
- [K2 Horizon 7B FP8 model card](https://huggingface.co/IFM/K2-Horizon-7B-FP8)
- Support: [support@ifm.ai](mailto:support@ifm.ai)

## License

The example code is licensed under [Apache License 2.0](LICENSE). Refer to each model’s model card and Marketplace terms for its license.
