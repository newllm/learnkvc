#!/usr/bin/env python3
"""在本地 Ollama 上对比 LLM 的代码准确率 / 工具调用 / 速度。

用法：
  python3 macos/ollama/eval_models.py
  python3 macos/ollama/eval_models.py --models qwen3.5:4b,qwen3.5:9b --ctx 32768
  python3 macos/ollama/eval_models.py --models qwen3.5:4b-q8_0 --tasks 8

说明：
  - 每题都带隐藏断言，LLM 只看到题目；生成的代码在子进程里跑断言，10s 超时。
  - 默认关闭 thinking（只对支持 think 的 LLM 发送 think:false，其他 LLM 自动跳过）。
  - 每个 LLM 跑完会做一次 write_file 工具调用测试，检查 tool_calls 是否非空。
  - 结果只是小样本参考：换成你自己仓库里的真实任务（带测试）更有意义。
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:11434"

TASKS = [
    {
        "name": "is_palindrome",
        "prompt": "写一个 Python 函数 is_palindrome(s: str) -> bool：忽略大小写和非字母数字字符后判断回文。只输出代码。",
        "tests": """
assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome("") is True
assert is_palindrome("0P") is False
""",
    },
    {
        "name": "flatten",
        "prompt": "写一个 Python 函数 flatten(x)：把任意嵌套的 list/tuple 递归展平成一个新 list，保持顺序。只输出代码。",
        "tests": """
assert flatten([1, [2, [3, (4, 5)], 6], 7]) == [1, 2, 3, 4, 5, 6, 7]
assert flatten([]) == []
assert flatten([[], [[]]]) == []
""",
    },
    {
        "name": "top_k_frequent",
        "prompt": "写一个 Python 函数 top_k_frequent(nums: list[int], k: int) -> list[int]：返回出现频率最高的 k 个元素，频率相同按数值升序。只输出代码。",
        "tests": """
assert top_k_frequent([1,1,1,2,2,3], 2) == [1,2]
assert top_k_frequent([4,4,4,4], 1) == [4]
assert top_k_frequent([5,3,3,5,5,3,7], 2) == [3,5]
""",
    },
    {
        "name": "parse_duration",
        "prompt": "写一个 Python 函数 parse_duration(s: str) -> int：把 '1h30m'、'45s'、'2h' 解析成秒；非法格式抛 ValueError。只输出代码。",
        "tests": """
assert parse_duration("1h30m") == 5400
assert parse_duration("45s") == 45
assert parse_duration("2h") == 7200
for bad in ["", "1x", "h"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
""",
    },
    {
        "name": "merge_intervals",
        "prompt": "写一个 Python 函数 merge_intervals(intervals: list[list[int]]) -> list[list[int]]：合并重叠区间，返回按起点排序的区间。只输出代码。",
        "tests": """
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
""",
    },
    {
        "name": "chunk_list",
        "prompt": "写一个 Python 函数 chunk_list(items: list, n: int) -> list[list]：按长度 n 切块，最后一块可以不满；n<=0 抛 ValueError。只输出代码。",
        "tests": """
assert chunk_list([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunk_list([], 3) == []
try:
    chunk_list([1], 0)
except ValueError:
    pass
else:
    raise AssertionError("n<=0 should raise")
""",
    },
]


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(model, prompt, ctx, think=True, max_tokens=800):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个 Python 编程助手，只输出代码，不要 markdown 代码围栏。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_ctx": ctx, "num_predict": max_tokens, "temperature": 0},
    }
    if think:
        payload["think"] = False
    try:
        return post("/api/chat", payload)
    except urllib.error.HTTPError as e:
        if think and e.code == 400:
            payload.pop("think", None)
            return post("/api/chat", payload)
        raise


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()


def run_tests(code, tests):
    try:
        p = subprocess.run(
            [sys.executable, "-c", code + "\n" + tests],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if p.returncode == 0:
        return True, ""
    return False, (p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout).strip() else "failed"


def tool_test(model, ctx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "创建 hello.py 打印 hello，必须调用 write_file 工具。"}],
        "stream": False,
        "options": {"num_ctx": ctx, "num_predict": 200, "temperature": 0},
        "tools": [{
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        }],
    }
    try:
        payload["think"] = False
        d = post("/api/chat", payload)
    except urllib.error.HTTPError:
        payload.pop("think", None)
        d = post("/api/chat", payload)
    return bool(d.get("message", {}).get("tool_calls"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5:4b,qwen3.5:9b", help="逗号分隔的 LLM 名")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tasks", type=int, default=len(TASKS), help=f"跑前 N 题（共 {len(TASKS)}）")
    args = ap.parse_args()

    tasks = TASKS[: max(1, min(args.tasks, len(TASKS)))]
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model}（num_ctx={args.ctx}, {len(tasks)} 题） ===")
        passed = 0
        times = []
        for i, task in enumerate(tasks, 1):
            t0 = time.time()
            try:
                d = chat(model, task["prompt"], args.ctx)
                code = extract_code(d.get("message", {}).get("content", ""))
                ok, err = run_tests(code, task["tests"])
            except Exception as e:  # noqa: BLE001
                ok, err = False, f"request failed: {e}"
            times.append(time.time() - t0)
            passed += int(ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] {task['name']:<16} {times[-1]:5.1f}s {err}")
        print(f"  pass@1 = {passed}/{len(tasks)}  avg={sum(times)/len(times):.1f}s")
        print(f"  tool_calls = {'OK' if tool_test(model, args.ctx) else 'MISSING'}")


if __name__ == "__main__":
    main()
