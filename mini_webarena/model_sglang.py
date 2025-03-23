# tokenizers.py
from typing import Any
from transformers import AutoTokenizer  # type: ignore

class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        # if provider == "openai":
        #     self.tokenizer = tiktoken.encoding_for_model(model_name)
        # elif provider == "huggingface":
        if provider == "huggingface":
            # print(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        # elif provider == "ours":
        #     self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

# Adapt from sglang in soup
import requests
import torch
import os

from typing import Any
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import time

# # 加载 SentenceTransformer 模型
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# def bert_reward(sentence_a, sentence_b):
#     """
#     计算两个句子的相似度（余弦相似度）。
#     """
#     embeddings_a = model.encode(sentence_a, convert_to_tensor=True).cpu()
#     embeddings_b = model.encode(sentence_b, convert_to_tensor=True).cpu()
#     similarity = 1 - cosine(embeddings_a, embeddings_b)
#     return similarity

def call_llm(
        lm_config,
        prompt,
) -> str:
    response: str
    port = os.getenv("PORT", 8000)  # 获取环境变量 PORT，默认 8000
    if lm_config == None:
        lm_config = {
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "gen_config": {
                "temperature": 0.0,
                "max_new_tokens": 4096
            }
        }
        # import SimpleNamespace
        from types import SimpleNamespace
        lm_config = SimpleNamespace(**lm_config)
    if lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        error_times = 0
        while error_times < 10:
            try:
                # 直接实现 call_refModel 的逻辑
                url = f"http://localhost:{port}/v1/chat/completions"
                data = {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": lm_config.gen_config.get("temperature", 0.0),
                    "max_tokens": lm_config.gen_config.get("max_new_tokens", 4096),
                }

                # 如果 lm_config 中有 top_p，则添加到请求中
                if "top_p" in lm_config.gen_config:
                    data["top_p"] = lm_config.gen_config["top_p"]

                # 发送请求
                api_response = requests.post(url, json=data)
                api_response.raise_for_status()
                json_data = api_response.json()
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                print(f"Error: {e}")
                error_times += 1
                print(f"Retrying ({error_times}/10)...")
                time.sleep(10)
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )
    return response

if __name__ == "__main__":
    input_text = "你好！"
    import os
    port = os.getenv("PORT", 8000)
    print(call_llm(lm_config=None, prompt=input_text))
