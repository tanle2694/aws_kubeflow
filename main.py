import kfp
import argparse

from kfp import dsl
from kfp.dsl._pipeline import PipelineConf
from kubernetes.client.models.v1_toleration import V1Toleration
from get_dex_cookie import get_user_auth_session_cookie


def cpu_op():
    cpu_op = dsl.ContainerOp(
        name='cpu-op',
        image='ubuntu:16.04',
        command=['echo', 'Hello from CPU node']
    )
    return cpu_op


def gpu_op():
    gpu_op = dsl.ContainerOp(
        name='gpu-op',
        image='nvcr.io/nvidia/pytorch:21.10-py3',
        command=['sh', '-c'],
        arguments=['nvidia-smi & sleep 60']
    ).set_gpu_limit(1)
    gpu_op.add_node_selector_constraint(label_name="nvidia.com/gpu", value="true")
    gpu_op.add_toleration(V1Toleration(effect="NoSchedule", key="nvidia.com/gpu", operator="Exists"))
    return gpu_op


@dsl.pipeline(
    name='aws kubeflow demo',
    description='Using bot cpu and gpu node with autoscale from zero'
)
def sample_pipeline():
    cpu_step = cpu_op()

    gpu_step_1 = gpu_op()
    gpu_step_1.after(cpu_step)

    gpu_step_2 = gpu_op()
    gpu_step_2.after(cpu_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeflow_host", type=str, help="URL to kubeflow server")
    parser.add_argument("--username", type=str, help='username to login kubeflow dex')
    parser.add_argument("--password", type=str, help='password to login kubeflow dex')
    parser.add_argument("--namespace", type=str, help='your namespace')

    args = parser.parse_args()

    endpoint = args.kubeflow_host
    api_username = args.username
    api_password = args.password
    api_endpoint = f"{endpoint}/pipeline"

    cookie = get_user_auth_session_cookie(api_endpoint, api_username, api_password)
    client = kfp.Client(host=api_endpoint,
                        cookies=cookie)
    pipeline_conf = PipelineConf()

    client.create_run_from_pipeline_func(sample_pipeline,
                                         experiment_name="aws kubeflow autoscale sample",
                                         namespace=args.namespace,
                                         pipeline_conf=pipeline_conf,
                                         arguments={}
                                         )