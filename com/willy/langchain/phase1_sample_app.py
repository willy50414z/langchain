from __future__ import annotations

import json
import sys

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
        print("Please set [deepseek] api_key in application.ini (repo root).")
        sys.exit(1)
