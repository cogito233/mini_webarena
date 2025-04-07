from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
import os

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def start_server():
    """
    First install the environment seperately;
        conda create -n sglang python=3.10 -y
        conda activate sglang
        pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
    """

    server_process, port = launch_server_cmd(
        """python -m sglang.launch_server \
    --model-path /home/zhiheng/cogito/base_models/qwen2.5-0.5b-wiki \
    --host 0.0.0.0
""",
    )
    wait_for_server(f"http://localhost:{port}")
    print_highlight(f"Server launched on port: {port}")
    return server_process, port


def stop_server(server_process):
    terminate_process(server_process)
    print_highlight("Server process terminated.")


try:
    server_process, port = start_server()
    print("Server started successfully on port:", port)
    # sleep until being killed
    import time
    while True:
        print("Server is running on port:", port)
        time.sleep(10)
finally:
    stop_server(server_process)