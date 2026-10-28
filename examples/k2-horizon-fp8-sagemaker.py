"""K2 Horizon 7B FP8 SageMaker package deployment and inference example.

Uses boto3's credential chain. No keys are embedded. A deployment starts billed
compute; run cleanup when finished. No external tool service is executed.
"""
import json
import re
import time
import uuid
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def request(messages, **kwargs):
    payload=dict(model='k2-horizon-7b', messages=messages, max_tokens=16384,
                 temperature=1.0, top_p=0.95,
                 chat_template_kwargs={'reasoning_effort':'high','tool_call_format':'xml'})
    payload.update(kwargs)
    return payload


def replay(message):
    history={key:message[key] for key in ['role','content','tool_calls'] if key in message}
    history['reasoning_content']=message.get('reasoning_content',message.get('reasoning','')) or ''
    return history


def invoke(runtime, endpoint, payload):
    response=runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json',
                                    Accept='application/json', Body=json.dumps({**payload,'stream':False}).encode())
    body=json.loads(response['Body'].read())
    choice=body['choices'][0]
    if choice['finish_reason']=='length':
        raise RuntimeError('Output budget exhausted; this is not a complete answer. Review prompt/output budgets.')
    return body


def stream(runtime, endpoint, payload):
    response=runtime.invoke_endpoint_with_response_stream(EndpointName=endpoint,
        ContentType='application/json',Accept='text/event-stream',Body=json.dumps({**payload,'stream':True,'stream_options':{'include_usage':True}}).encode())
    parts=[]
    for event in response['Body']:
        if 'PayloadPart' not in event:raise RuntimeError('Streaming error: '+str(event))
        parts.append(event['PayloadPart']['Bytes'])
    # Decode after joining: UTF-8 characters and SSE records may cross AWS chunks.
    text=b''.join(parts).decode('utf-8').replace('\r\n','\n')
    message={'role':'assistant','content':'','reasoning_content':''};calls={};finish=None;done=False;usage=None
    for block in text.split('\n\n'):
        data='\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith('data:'))
        if not data:continue
        if done:raise RuntimeError('Unexpected data after stream completion')
        if data=='[DONE]':done=True;continue
        chunk=json.loads(data)
        if 'error' in chunk:raise RuntimeError(str(chunk['error']))
        if chunk.get('usage') is not None:usage=chunk['usage']
        for choice in chunk.get('choices',[]):
            delta=choice.get('delta',{})
            message['content']+=delta.get('content') or ''
            message['reasoning_content']+=delta.get('reasoning_content',delta.get('reasoning')) or ''
            for call in delta.get('tool_calls',[]):
                target=calls.setdefault(call['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                target['id']+=call.get('id') or ''
                for key in ['name','arguments']:target['function'][key]+=call.get('function',{}).get(key) or ''
            finish=choice.get('finish_reason') or finish
    if not done or not finish:raise RuntimeError('Incomplete response stream')
    if finish=='length':raise RuntimeError('Stream exhausted its output budget before a complete answer')
    if calls:message['tool_calls']=[calls[i] for i in sorted(calls)]
    return {'message':message,'finish_reason':finish,'usage':usage}


def demonstrate(session, endpoint):
    def check_answer(message, expected):
        assert re.search(r'(?<!\d)'+str(expected)+r'(?!\d)',message.get('content') or ''), message
    runtime=session.client('sagemaker-runtime',config=Config(read_timeout=300,retries={'max_attempts':0}))
    first=[{'role':'user','content':'What is 6 + 1? Answer briefly.'}]
    chat=invoke(runtime,endpoint,request(first))
    assert chat['choices'][0]['finish_reason']=='stop'
    check_answer(chat['choices'][0]['message'],7)
    history=first+[replay(chat['choices'][0]['message']),{'role':'user','content':'Double your previous answer. Answer briefly.'}]
    followup=stream(runtime,endpoint,request(history))
    assert followup['finish_reason']=='stop'
    check_answer(followup['message'],14)
    history+=[replay(followup['message']),{'role':'user','content':'Add 3 to that result. Answer briefly.'}]
    third=invoke(runtime,endpoint,request(history))
    assert third['choices'][0]['finish_reason']=='stop'
    check_answer(third['choices'][0]['message'],17)
    tools=[{'type':'function','function':{'name':'get_inventory','description':'Look up available product quantity.',
        'parameters':{'type':'object','properties':{'sku':{'type':'string'}},'required':['sku'],'additionalProperties':False}}}]
    messages=[{'role':'system','content':'Use the inventory tool for stock quantities. Do not guess inventory levels.'},
              {'role':'user','content':'How many units of SKU DEMO-7B are available?'}]
    tool=stream(runtime,endpoint,request(messages,tools=tools,tool_choice='auto'))
    assert tool['finish_reason']=='tool_calls'
    call,=tool['message']['tool_calls']
    assert call['id'], 'A tool call must have an ID for result replay'
    assert call['function']['name']=='get_inventory' and json.loads(call['function']['arguments'])=={'sku':'DEMO-7B'}
    # Synthetic fixture only. Applications must authorize/validate real tools.
    messages+=[replay(tool['message']),{'role':'tool','tool_call_id':call['id'],
               'content':json.dumps({'sku':'DEMO-7B','available_quantity':17,'source':'synthetic-example'})}]
    answer=invoke(runtime,endpoint,request(messages,tools=tools,tool_choice='auto'))
    assert answer['choices'][0]['finish_reason']=='stop'
    check_answer(answer['choices'][0]['message'],17)
    return {'chat':chat,'streamed_followup':followup,'history_replay':third,'streamed_tool_call':tool,'tool_result':answer}


def long_document_demo(session, endpoint):
    runtime=session.client('sagemaker-runtime',config=Config(read_timeout=300,retries={'max_attempts':0}))
    document=('hello '*6000+'\nThe shipment verification code is SAGE-842.\n'+'hello '*6000)
    payload=request([{'role':'user','content':
        'Find the shipment verification code in this synthetic document. Return only the code.\n'+document}],
        max_tokens=16384,seed=7)
    response=invoke(runtime,endpoint,payload)
    choice=response['choices'][0]
    assert choice['finish_reason']=='stop' and 'SAGE-842' in choice['message']['content']
    assert 12000 <= response['usage']['prompt_tokens'] < 15360, response['usage']
    return response


PROFILES = {
    '32k': {'instance_type': 'ml.g6.2xlarge', 'context_tokens': 32768, 'tensor_parallel': 1},
    '64k': {'instance_type': 'ml.g6.12xlarge', 'context_tokens': 65536, 'tensor_parallel': 4},
}


def deploy(session, package_arn, role_arn, profile='32k', name=None, resources=None):
    if profile not in PROFILES:
        raise ValueError('Choose the 32k or 64k profile')
    recipe=PROFILES[profile]
    sm=session.client('sagemaker');package=sm.describe_model_package(ModelPackageName=package_arn)
    if package['ModelPackageStatus']!='Completed':raise RuntimeError('Package is not ready')
    if recipe['instance_type'] not in package['InferenceSpecification']['SupportedRealtimeInferenceInstanceTypes']:
        raise RuntimeError('Package does not support the selected instance type')
    name=name or 'k2-horizon-7b-fp8-example-'+uuid.uuid4().hex[:8]
    resources={} if resources is None else resources
    if resources:raise ValueError('Pass an empty resource dictionary for a new deployment')
    resources.update(model=name,config=name+'-'+profile,endpoint=name+'-test')
    print('Keep these names for cleanup:',json.dumps(resources),flush=True)
    try:
        sm.create_model(ModelName=name,ExecutionRoleArn=role_arn,EnableNetworkIsolation=True,
            PrimaryContainer={'ModelPackageName':package_arn,'Environment':{'K2_SERVED_MODEL_NAME':'k2-horizon-7b',
                'K2_TENSOR_PARALLEL_SIZE':str(recipe['tensor_parallel']),
                'K2_MAX_MODEL_LEN':str(recipe['context_tokens']),'K2_MAX_NUM_SEQS':'1',
                'K2_MAX_NUM_BATCHED_TOKENS':'512','K2_ENFORCE_EAGER':'1',
                'K2_GPU_MEMORY_UTILIZATION':'0.90','VLLM_ENABLE_CUDA_COMPATIBILITY':'0',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}})
        resources['model_created']=True
        sm.create_endpoint_config(EndpointConfigName=resources['config'],ProductionVariants=[{
            'VariantName':'AllTraffic','ModelName':name,'InstanceType':recipe['instance_type'],'InitialInstanceCount':1,
            'InitialVariantWeight':1.0,'InferenceAmiVersion':'al2023-ami-sagemaker-inference-gpu-4-1',
            'ModelDataDownloadTimeoutInSeconds':1800,'ContainerStartupHealthCheckTimeoutInSeconds':900}])
        resources['config_created']=True
        # Newly created configurations can take a short time to become visible.
        for attempt in range(12):
            try:
                sm.create_endpoint(EndpointName=resources['endpoint'],EndpointConfigName=resources['config'])
                break
            except ClientError as error:
                message=error.response['Error'].get('Message','').lower()
                if (attempt == 11 or error.response['Error']['Code'] != 'ValidationException'
                        or 'endpoint configuration' not in message or 'could not find' not in message):
                    raise
                time.sleep(5)
        resources['endpoint_created']=True
        for _ in range(80):
            state=sm.describe_endpoint(EndpointName=resources['endpoint'])
            print('Endpoint:',state['EndpointStatus'],flush=True)
            if state['EndpointStatus']=='InService':return resources
            if state['EndpointStatus']=='Failed':raise RuntimeError(state.get('FailureReason','Deployment failed'))
            time.sleep(30)
        raise TimeoutError('Deployment did not finish within 40 minutes')
    except BaseException:
        cleanup(session,resources)
        raise


def absent_resource(error):
    detail=error.response['Error']
    return detail['Code']=='ValidationException' and any(
        phrase in detail['Message'].lower() for phrase in ('could not find','does not exist','not found'))


def cleanup(session, resources):
    sm=session.client('sagemaker')
    if resources.get('endpoint_created'):
        # SageMaker rejects deletion while endpoint creation is in progress.
        # Keep the resource markers intact if this bounded wait expires.
        for _ in range(60):
            try:state=sm.describe_endpoint(EndpointName=resources['endpoint'])
            except ClientError as error:
                if absent_resource(error):
                    state=None
                    break
                raise
            if state['EndpointStatus']!='Creating':break
            print('Cleanup waiting for SageMaker creation lock:',resources['endpoint'],flush=True)
            time.sleep(15)
        else:raise RuntimeError('SageMaker still holds the creation lock; retry cleanup and inspect AWS before leaving')
        if state is not None and state['EndpointStatus']!='Deleting':
            sm.delete_endpoint(EndpointName=resources['endpoint'])
        for _ in range(40):
            try:sm.describe_endpoint(EndpointName=resources['endpoint'])
            except ClientError as error:
                if absent_resource(error):break
                raise
            time.sleep(15)
        else:raise RuntimeError('Endpoint deletion not confirmed; inspect the AWS console before leaving')
        resources['endpoint_created']=False
    if resources.get('config_created'):
        try:sm.delete_endpoint_config(EndpointConfigName=resources['config'])
        except ClientError as error:
            if not absent_resource(error):raise
        resources['config_created']=False
    if resources.get('model_created'):
        try:sm.delete_model(ModelName=resources['model'])
        except ClientError as error:
            if not absent_resource(error):raise
        resources['model_created']=False
    print('Example resources cleaned up.',flush=True)


def batch_demo(session, package_arn, role_arn, bucket, resources, name=None):
    """Run one generated JSON request on a temporary 32K batch model."""
    if resources:raise ValueError('Pass an empty resource dictionary for a new batch job')
    sm=session.client('sagemaker');s3=session.client('s3')
    name=name or 'k2-horizon-7b-fp8-batch-'+uuid.uuid4().hex[:8]
    prefix='k2-horizon-examples/'+name
    resources.update(model=name,job=name,bucket=bucket,input_key=prefix+'/input/request.json',
                     output_key=prefix+'/output/request.json.out')
    print('Keep these names for cleanup:',json.dumps(resources),flush=True)
    try:
        sample=request([{'role':'user','content':'Calculate 23 times 47. Return only the integer result.'}],stream=False)
        s3.put_object(Bucket=bucket,Key=resources['input_key'],Body=json.dumps(sample).encode(),ContentType='application/json')
        resources['input_created']=True
        sm.create_model(ModelName=name,ExecutionRoleArn=role_arn,EnableNetworkIsolation=True,
            PrimaryContainer={'ModelPackageName':package_arn,'Environment':{
                'K2_SERVED_MODEL_NAME':'k2-horizon-7b','K2_MAX_MODEL_LEN':'32768','K2_TENSOR_PARALLEL_SIZE':'1',
                'K2_MAX_NUM_SEQS':'1','K2_MAX_NUM_BATCHED_TOKENS':'512','K2_ENFORCE_EAGER':'1',
                'K2_GPU_MEMORY_UTILIZATION':'0.90','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
                'VLLM_ENABLE_CUDA_COMPATIBILITY':'1','VLLM_CUDA_COMPATIBILITY_PATH':'/usr/local/cuda-13.0/compat',
                'LD_LIBRARY_PATH':'/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib'}})
        resources['model_created']=True
        sm.create_transform_job(TransformJobName=name,ModelName=name,MaxConcurrentTransforms=1,MaxPayloadInMB=1,
            ModelClientConfig={'InvocationsTimeoutInSeconds':300,'InvocationsMaxRetries':0},
            TransformInput={'DataSource':{'S3DataSource':{'S3DataType':'S3Prefix','S3Uri':'s3://'+bucket+'/'+resources['input_key']}},
                            'ContentType':'application/json','CompressionType':'None','SplitType':'None'},
            TransformOutput={'S3OutputPath':'s3://'+bucket+'/'+prefix+'/output/', 'Accept':'application/json','AssembleWith':'None'},
            TransformResources={'InstanceType':'ml.g6.2xlarge','InstanceCount':1,'TransformAmiVersion':'al2-ami-sagemaker-batch-gpu-535'})
        resources['job_created']=True
        for _ in range(80):
            state=sm.describe_transform_job(TransformJobName=name)
            print('Batch job:',state['TransformJobStatus'],flush=True)
            if state['TransformJobStatus']=='Completed':
                response=json.loads(s3.get_object(Bucket=bucket,Key=resources['output_key'])['Body'].read())
                choice=response['choices'][0]
                if choice['finish_reason']!='stop':raise RuntimeError('Batch response is incomplete: '+str(choice['finish_reason']))
                if choice['message']['content'].strip()!='1081':raise RuntimeError('Unexpected answer for the arithmetic example')
                return {'input':sample,'output':response}
            if state['TransformJobStatus'] in ('Failed','Stopped'):
                raise RuntimeError(state.get('FailureReason','Batch job stopped'))
            time.sleep(30)
        raise TimeoutError('Batch job did not finish within 40 minutes')
    finally:
        cleanup_batch(session,resources)


def cleanup_batch(session, resources):
    """Stop active work and remove only this example's model and exact S3 keys."""
    sm=session.client('sagemaker')
    if resources.get('job_created'):
        state=sm.describe_transform_job(TransformJobName=resources['job'])
        if state['TransformJobStatus']=='InProgress':
            sm.stop_transform_job(TransformJobName=resources['job'])
        for _ in range(60):
            state=sm.describe_transform_job(TransformJobName=resources['job'])
            if state['TransformJobStatus'] in ('Completed','Failed','Stopped'):break
            time.sleep(15)
        else:raise RuntimeError('Batch termination not confirmed; retain resources and retry cleanup_batch')
        resources['job_created']=False
    if resources.get('model_created'):
        try:sm.delete_model(ModelName=resources['model'])
        except ClientError as error:
            if not absent_resource(error):raise
        resources['model_created']=False
    if resources.get('input_created'):
        s3=session.client('s3')
        for key in ('input_key','output_key'):
            s3.delete_object(Bucket=resources['bucket'],Key=resources[key])
        resources['input_created']=False
    print('Batch compute stopped and example resources cleaned up.',flush=True)
