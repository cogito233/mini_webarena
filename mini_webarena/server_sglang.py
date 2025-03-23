# model_server.py

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
import os

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'  # 设置指定显卡

# 如果在 CI 环境中需要用 patch，非 CI 则从 sglang.utils 里导入
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def start_server():
    """
    启动一个基于 sglang.launch_server 的 LLM 服务，返回(server_process, port)
    这里以 meta-llama/Meta-Llama-3.1-8B-Instruct 为例。
    """
    server_process, port = launch_server_cmd(
        """python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0
""",
    )
    # 等待服务启动完成
    wait_for_server(f"http://localhost:{port}")
    print_highlight(f"Server launched on port: {port}")
    return server_process, port


def stop_server(server_process):
    """
    测试完成后关闭该进程。
    """
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