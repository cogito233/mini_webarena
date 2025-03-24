# import nest_asyncio
# nest_asyncio.apply()
# import asyncio
import re
import random
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# 假设你的 BaseLanguageBasedEnv 就在同目录下的 base.py
from .env_base import BaseLanguageBasedEnv
from .browser_actions import (
    Action,
    ActionTypes,
    create_stop_action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from .browser_env import ScriptBrowserEnv, Trajectory


class WikiQAEnv(BaseLanguageBasedEnv):
    def __init__(
            self,
            question, gt,
            max_steps: int = 10,
            threshold: float = 0.7,
            prompt_format="last",  # full, last, single, tunc
            url = None
    ):
        """
        Args:
            dataset: 包含多条问答数据的 DataFrame, 假设有列 'question' 和 'answer'
            seed: 用来选取第几条问答
            max_steps: 最大交互步数，超限后强制结束
            threshold: 当 final answer 与真实答案的相似度 >= threshold 则视为成功
        """

        # print("=" * 30)
        # print("WikiQAEnv init")

        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        self.prompt_format = prompt_format
        # print("Init prompt_format", prompt_format, self.prompt_format)
        # print("[DEBUG] WikiQAEnv init Checkpoint 1")

        self.current_step = 0
        self.done = False
        self.reward = 0.0
        # print("[DEBUG] WikiQAEnv init Checkpoint 2")
        # get the dict of the dataset {seed}-th row from pd.df
        self.question = question
        self.gt = gt
        # print("[DEBUG] WikiQAEnv init Checkpoint 3")

        self.pred = None
        # TODO, heed to change to multiple answers
        self.answer_similarity = 0.0
        self.answer_made = False  # 是否已做出回答
        self.obs_modality = "text"

        from .create_dataset import TEMPLATES
        self.template_dict = TEMPLATES['qwen-instruct']
        # print("[DEBUG] WikiQAEnv init Checkpoint 4")

        # Web Browser Environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
            save_trace_enabled=True,
            sleep_after_execution=0.0,
            simple_mode=True
        )

        # print("[DEBUG] WikiQAEnv init Checkpoint 5")
        from .agent import construct_promptConstructor
        self.prompt_constructor, self.tokenizer, _ = construct_promptConstructor("Qwen/Qwen2.5-14B-Instruct", None)
        if url == None:
            self.url = "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        else:
            self.url = url

        # print("[DEBUG] WikiQAEnv init Checkpoint 6")
        obs, info = self.env.reset_without_config(start_url=self.url)

        # print("[DEBUG] WikiQAEnv init Checkpoint 7")
        self.history = [{"role": "system"}, {"role": "user", "question": self.question, "url": self.url,
                                             "observation": obs[self.obs_modality], "previous_action": None}]

        # print("[DEBUG] WikiQAEnv init Checkpoint 8")
        # 重置跟踪变量
        self._reset_tracking_variables()

        # print("=" * 30)

    def reset_qa(self, question, gt, url=None) -> Any:
        print("[DEBUG] WikiQAEnv reset_qa Checkpoint 1")
        self.question = question
        self.gt = gt

        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False
        self.current_step = 0
        self.done = False

        if url is not None:
            self.url = url
        else:
            self.url = "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

        obs, info = self.env.reset_without_config(start_url=self.url)

        print("[DEBUG] WikiQAEnv reset_qa Checkpoint 2")
        self.history = [{"role": "system"}, {"role": "user", "question": self.question, "url": self.url,
                                             "observation": obs[self.obs_modality], "previous_action": None}]

        self._reset_tracking_variables()
        return self.render()

    def __str__(self):
        return f"WikiQAEnv(seed={self.seed}, question={self.question}, answer={self.gt})"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Returns:
            observation (str): 当前环境渲染
            reward (float): 本step得到的奖励
            done (bool): 是否结束
            info (dict): 额外信息
        """
        if self.done:
            return (self.render(), 0.0, True, {"action_is_effective": False})
        # TODO
        # Step 1. check if done, calculate reward, append history
        action_extracted, action_str = self.extract_action(action)
        # from .rl_utils import format_score
        # from .evaluator import fuzzy_match
        # is_success = action_extracted["action_type"] == ActionTypes.NONE
        format_reward = 0  # Placeholder
        if action_extracted["action_type"] == ActionTypes.CHECK or action_extracted["action_type"] == ActionTypes.STOP:
            self.answer_made = True
            self.done = True
            # self.pred = action_extracted["raw_prediction"]
            self.pred = action_extracted['answer']
            # print(action_extracted)
            # print("#"*100)
            self.answer_similarity = 0  # Modified, no need calculate reward here
            answer_reward = self.answer_similarity
            reward = -0.01 + format_reward + answer_reward
            self.reward += reward
            obs = self.render()
            self.history.append(
                {"role": "assistant", "pred": action, "reward": reward, "action_extracted": action_extracted})
            # print(f"Format Reward: {format_reward}, Answer Reward: {answer_reward}, Total Reward: {reward}")
            return (obs, reward, self.done, {"action_is_effective": True})

        reward = -0.01 + format_reward
        self.reward += reward
        self.history.append(
            {"role": "assistant", "pred": action, "reward": reward, "action_extracted": action_extracted})
        # Step 2. execute action
        # print(action_extracted)
        try:
            obs, _, terminated, _, info = self.env.step(action_extracted)
        except Exception as e:
            print("######################### Error in run step")
            print(self.render())
            print("######################### Error in run step")
            raise e
        self.current_step += 1
        self.done = self.current_step >= self.max_steps or terminated
        # Step 3. add observation to history
        # print(obs)
        # print(terminated)
        self.url = self.env.page.url
        self.history.append(
            {"role": "user", "question": self.question, "url": self.url, "observation": obs[self.obs_modality],
             "previous_action": action_str})
        observation = self.render()
        # print()
        # print(observation)
        # exit(0)
        # Step 4. check if meet the break condition
        if self.check_break_condition():
            self.done = True
        return (observation, reward, self.done, info)

    def finished(self) -> bool:
        return self.done

    def render(self, prompt_format=None) -> str:

        if prompt_format is None:
            prompt_format = self.prompt_format

        # print("prompt_format", prompt_format, self.prompt_format)

        if prompt_format == "full":
            ans = ""
            # print("#"*100)
            # print(len(self.history))
            for item in self.history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective=item["question"], url=item["url"], observation
                    =item["observation"], previous_action=item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        elif prompt_format == "single":
            history = [self.history[0], self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective=item["question"], url=item["url"], observation
                    =item["observation"], previous_action=item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        elif prompt_format == "last":
            history = [self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective=item["question"], url=item["url"], observation
                    =item["observation"], previous_action=item["previous_action"])
                else:
                    print(item)
                    raise ValueError("role not recognized")
            return ans
        else:
            raise NotImplementedError

    def get_state(self):
        """
        导出当前环境的状态，不包括不可序列化的浏览器对象（self.env）。
        保存问答状态、交互历史以及浏览器关键信息（当前页面 URL 与 storage_state）。
        """
        # print(self.max_steps)
        # print("##########################")
        state = {
            "question": self.question,
            "gt": self.gt,
            "max_steps": self.max_steps,
            "threshold": self.threshold,
            "prompt_format": self.prompt_format,
            "current_step": self.current_step,
            "done": self.done,
            "reward": self.reward,
            "pred": self.pred,
            "answer_similarity": self.answer_similarity,
            "answer_made": self.answer_made,
            "obs_modality": self.obs_modality,
            "history": self.history,
            "template_dict": self.template_dict,
            "url": self.url,
            # 如果 prompt_constructor 和 tokenizer 是可序列化的，也保存它们（或后续可根据需要重建）
            "prompt_constructor": self.prompt_constructor,
            "tokenizer": self.tokenizer,
        }
        # 保存浏览器部分关键信息
        browser_state = {}
        if hasattr(self.env, "page") and self.env.page is not None:
            browser_state["page_url"] = self.env.page.url
        if hasattr(self.env, "context") and self.env.context is not None:
            try:
                # 利用 Playwright 提供的 storage_state() 获取 cookies、localStorage 等信息
                browser_state["storage_state"] = self.env.context.storage_state()
            except Exception as e:
                browser_state["storage_state"] = None
        state["browser_state"] = browser_state
        # print(state)
        return state

    def close(self):
        """
        关闭当前环境，释放浏览器资源。
        注意：调用 self.env.close() 会调用 ScriptBrowserEnv.close()，
        其内部会调用 context_manager.__exit__() 来关闭 Playwright 连接。
        """
        try:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
        except Exception as e:
            print("Error closing environment:", e)

    def load_state(self, state):
        """
        根据保存的 state 更新当前环境的状态，不依赖 ScriptBrowserEnv.reset_without_config()。
        手动重建浏览器环境，恢复保存的 page_url 与 storage_state。
        """
        # 恢复业务状态字段
        # print(state)
        self.question = state.get("question", self.question)
        self.gt = state.get("gt", self.gt)
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self.threshold = state.get("threshold", self.threshold)
        self.prompt_format = state.get("prompt_format", self.prompt_format)
        self.current_step = state.get("current_step", self.current_step)
        self.done = state.get("done", self.done)
        self.reward = state.get("reward", self.reward)
        self.pred = state.get("pred", self.pred)
        self.answer_similarity = state.get("answer_similarity", self.answer_similarity)
        self.answer_made = state.get("answer_made", self.answer_made)
        self.obs_modality = state.get("obs_modality", self.obs_modality)
        self.history = state.get("history", self.history)
        self.template_dict = state.get("template_dict", self.template_dict)
        self.url = state.get("url", self.url)
        self.prompt_constructor = state.get("prompt_constructor", self.prompt_constructor)
        self.tokenizer = state.get("tokenizer", self.tokenizer)

        # 恢复浏览器状态：提取保存的浏览器状态信息
        browser_state = state.get("browser_state", {})
        start_url = browser_state.get("page_url", self.url)
        storage_state = browser_state.get("storage_state", None)
        # 如果保存的 storage_state 是 dict，则写入临时 JSON 文件（reset 新环境时传入文件路径）
        import json
        from pathlib import Path
        from playwright.sync_api import sync_playwright
        if isinstance(storage_state, dict):
            tmp_file = Path("tmp_storage_state.json")
            tmp_file.write_text(json.dumps(storage_state))
            storage_state_param = str(tmp_file)
        else:
            storage_state_param = storage_state

        # 先尝试关闭当前环境
        try:
            self.env.close()
        except Exception as e:
            print("Error closing old environment:", e)

        # 手动重建浏览器环境（逻辑参考 ScriptBrowserEnv.reset_without_config）
        self.env.context_manager = sync_playwright()
        self.env.playwright = self.env.context_manager.start()
        self.env.browser = self.env.playwright.chromium.launch(
            headless=self.env.headless,
            slow_mo=self.env.slow_mo,
            args=["--no-sandbox"]
        )
        self.env.context = self.env.browser.new_context(
            viewport=self.env.viewport_size,
            storage_state=storage_state_param,
            geolocation=None,
            device_scale_factor=1,
        )
        if self.env.save_trace_enabled:
            self.env.context.tracing.start(screenshots=True, snapshots=True)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.env.context.new_page()
                client = page.context.new_cdp_session(page)
                if self.env.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client
                page.goto(url)
            self.env.page = self.env.context.pages[0]
            self.env.page.bring_to_front()
        else:
            self.env.page = self.env.context.new_page()
            client = self.env.page.context.new_cdp_session(self.env.page)
            if self.env.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            self.env.page.client = client
        self.env.reset_finished = True

    # ============== Tool Functions ==============
    def extract_action(self, response: str):
        force_prefix = self.prompt_constructor.instruction[
            "meta_data"
        ].get("force_prefix", "")
        response = f"{force_prefix}{response}"
        try:
            parsed_response = self.prompt_constructor.extract_action(
                response
            )
            # print(parsed_response)
            action = create_id_based_action(parsed_response)
            # print(action)
            # exit(0)
            action["raw_prediction"] = response
        except ActionParsingError as e:
            print(f"ActionParsingError: {e}")
            # raise e
            action = create_none_action()
            parsed_response = "The action is invalid, please retry"
        return action, parsed_response

    def check_break_condition(self):
        # If something happen in the history, then return -1
        # TODO
        return False

    def copy(self) -> 'BaseEnv':
        pass

    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        pass

    def success(self) -> bool:
        pass


# ============ 测试示例 ============
def test_wiki_qa_env():
    import time
    # 1. 实例化环境
    env = WikiQAEnv("Who is current US president", "Biden", prompt_format="single")

    # 2. Reset 环境，打印初始渲染结果
    print("=== Initial Reset & Render ===")
    observation = env.reset()
    print("Render:", env.render())

    # 保存状态后关闭当前环境
    print("=== Saving state after reset ===")
    state = env.get_state()
    env.close()
    del env

    # 新建环境并加载状态
    env = WikiQAEnv("Who is current US president", "Biden", prompt_format="single")
    env.load_state(state)
    print("=== After reloading state, render ===")
    print("Render:", env.render())

    # 3. 第一次 step: 执行 action_1
    action_1 = "<think>balabalabalabala</think>\n```click [99]```"
    print("=== Step 1: Action ===")
    print("Action:", action_1)
    observation, reward, done, info = env.step(action_1)
    print("=== Observation After Action 1 ===")
    print("Render:", env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")

    # 保存状态后关闭当前环境
    print("=== Saving state after Step 1 ===")
    state = env.get_state()
    env.close()
    del env

    # 新建环境并加载状态
    env = WikiQAEnv("Who is current US president", "Biden", prompt_format="single")
    env.load_state(state)
    print("=== After reloading state, render ===")
    print("Render:", env.render())

    # 4. 第二次 step: 执行 action_2
    action_2 = "<think>balabalabalabala</think>\n```stop [Sept 18, 2018]```"
    print("=== Step 2: Action ===")
    print("Action:", action_2)
    observation, reward, done, info = env.step(action_2)
    print("=== Observation After Action 2 ===")
    print("Render:", env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")

    # 保存状态后关闭当前环境
    print("=== Saving state after Step 2 ===")
    state = env.get_state()
    env.close()
    del env

    # 新建环境并加载状态
    env = WikiQAEnv("Who is current US president", "Biden", prompt_format="single")
    env.load_state(state)
    print("=== After reloading state, render ===")
    print("Render:", env.render())

    # 最后关闭环境
    env.close()


import pickle
from collections.abc import Mapping, Sequence, Set


def test_pickle_recursively(obj, prefix="obj", visited=None, max_depth=5):
    """
    尝试递归遍历 obj 的属性，并测试是否可被 pickle。
    prefix: 用于打印时表明当前对象层级的标签。
    visited: 避免重复检测同一个对象 (by id)。
    max_depth: 递归最大深度，防止无限循环。
    """

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return  # 已检测过这个对象，防重复循环
    visited.add(obj_id)

    # 1) 先尝试对当前对象做 pickle
    try:
        pickle.dumps(obj)
        print(f"[✅] {prefix} is picklable: {type(obj)}")
    except Exception as e:
        print(f"[❌] {prefix} cannot be pickled: {type(obj)}, reason: {e}")

        # 如果深度还够，继续探查内部属性(如果有)
        if max_depth > 0:
            # 2) 如果对象有 __dict__，递归其属性
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    test_pickle_recursively(v, f"{prefix}.{k}", visited, max_depth - 1)

            # 3) 如果对象是常见容器类型(list/tuple/dict/set)，也要继续遍历
            elif isinstance(obj, Mapping):
                for k, v in obj.items():
                    test_pickle_recursively(v, f"{prefix}[{k!r}]", visited, max_depth - 1)
            elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes, bytearray)):
                # list/tuple/set
                for i, v in enumerate(obj):
                    test_pickle_recursively(v, f"{prefix}[{i}]", visited, max_depth - 1)

        return  # 测完内部属性就可以返回

    # 只有在当前对象本身 pickle 成功后，才值得深入检查子属性
    # 因为如果本体都不可 pickle，就已经报错了
    if max_depth > 0:
        print("#" * 50)
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                test_pickle_recursively(v, f"{prefix}.{k}", visited, max_depth - 1)
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                test_pickle_recursively(v, f"{prefix}[{k!r}]", visited, max_depth - 1)
        elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes, bytearray)):
            for i, v in enumerate(obj):
                test_pickle_recursively(v, f"{prefix}[{i}]", visited, max_depth - 1)


def test_pickle():
    env = WikiQAEnv("Who is current US president", "Biden")
    print("=== Testing Picklability of WikiQAEnv Attributes ===")
    test_pickle_recursively(env)


if __name__ == "__main__":
    test_wiki_qa_env()
