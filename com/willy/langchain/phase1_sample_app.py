from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from com.willy.langchain.svc.langchain_svc import LangChainService


class Phase1SampleApp:
    """LangChain 學習計畫第一階段 sample 入口。"""

    def __init__(self) -> None:
        self.langchain_svc = LangChainService()

    def run(self) -> None:
        print("=== Phase 1 / Prompt Template ===")
        prompt_result = self.langchain_svc.demo_prompt_template(topic="LangChain Model I/O", weeks=2)
        print(prompt_result)

        print("\n=== Phase 1 / Chat Models vs LLMs ===")
        compare_result = self.langchain_svc.demo_chat_model_vs_llm(
            question="LangChain 的 Chat Model 與傳統 LLM 呼叫差異是什麼？"
        )
        print("[Chat Model]")
        print(compare_result["chat_model"])
        print("\n[LLM]")
        print(compare_result["llm"])

        print("\n=== Phase 1 / Output Parsers ===")
        json_result = self.langchain_svc.demo_json_output_parser(topic="Prompt Templates")
        print("[JSON Parser]")
        print(json.dumps(json_result, ensure_ascii=False, indent=2))

        pydantic_result = self.langchain_svc.demo_pydantic_output_parser(topic="Output Parsers")
        print("\n[Pydantic Parser]")
        print(pydantic_result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app = Phase1SampleApp()
    try:
        app.run()
    except EnvironmentError as exc:
        print(f"[Environment Error] {exc}")
        print("Please set [local_model] base_url / chat_model / llm_model in application.ini (repo root).")
        sys.exit(1)

    # 1. 有2種發送問題給模型的方式：
    #   1.1 透過chat model發送: 可以分開指定系統prompt("你是技術助理，回答限制在 80 字內。")和user的prompt("Chat Models & LLMs: 以 Gemini Chat API 展示 chat 與 llm-style prompt 差異。")
    #   1.2 透過llm發送: 只能指定一個prompt("你是技術助理，請用 80 字內回答。\n問題: {question}\n回答:")
    # 2. 有2種輸出解析器：
    #   2.1 json output parser: 將模型的輸出轉換為json格式
    #   2.2 pydantic output parser: 將模型的輸出轉換為自訂義物件，請模型填入
    # 3. 有2種chain：
    #   3.1 sequential chain: 串接多個處理步驟
    #   3.2 parallel chain: 並行處理多個步驟
    # 4. 有2種memory：
    #   4.1 chat message history: 儲存對話紀錄
    #   4.2 conversation buffer: 管理對話窗口的大小與摘要
