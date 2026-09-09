from __future__ import annotations

import os
import random
import re
import shutil
from typing import Final

from transitions.extensions import GraphMachine

DECISION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[DECISION:\s*(PASS|REJECT|NEEDS_HUMAN)\s*]", re.IGNORECASE)
RESULT_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[RESULT:\s*(SUCCESS|FAIL)\s*]", re.IGNORECASE)


class AIWorkflowController:
    states = [
        "idle",
        "gemini_thinking",
        "chatgpt_reviewing",
        "human_decision",
        "code_executing",
    ]

    transitions = [
        {"trigger": "start_project", "source": "idle", "dest": "gemini_thinking"},
        {"trigger": "submit_to_review", "source": "gemini_thinking", "dest": "chatgpt_reviewing"},
        {"trigger": "review_pass", "source": "chatgpt_reviewing", "dest": "code_executing"},
        {"trigger": "review_fail", "source": "chatgpt_reviewing", "dest": "gemini_thinking"},
        {"trigger": "needs_human", "source": "chatgpt_reviewing", "dest": "human_decision"},
        {"trigger": "human_override_pass", "source": "human_decision", "dest": "code_executing"},
        {"trigger": "human_give_new_prompt", "source": "human_decision", "dest": "gemini_thinking"},
        {"trigger": "execution_success", "source": "code_executing", "dest": "idle"},
        {"trigger": "execution_fail", "source": "code_executing", "dest": "gemini_thinking"},
    ]

    def __init__(self) -> None:
        self.machine = GraphMachine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial="idle",
            auto_transitions=False,
            title="AI Multi-Agent Workflow",
            show_conditions=True,
            after_state_change="on_state_change",
        )

    def on_state_change(self) -> None:
        print(f"[State Change] 當前狀態: {self.state}")

    def mock_call_gemini_cli(self) -> str:
        print("[System] 呼叫 Gemini CLI 生成代碼...")
        return (
            "```python\n"
            "def hello(name: str) -> str:\n"
            "    return f'Hello, {name}!'\n"
            "```\n"
        )

    def mock_call_chatgpt_cli(self) -> str:
        print("[System] 呼叫 ChatGPT CLI 進行代碼審查...")
        decision = random.choice(["PASS", "REJECT", "NEEDS_HUMAN"])
        return (
            "## Review Result\n"
            "- Style check complete.\n"
            "- Unit tests suggested.\n"
            f"[DECISION: {decision}]\n"
        )

    def mock_call_codex_executor(self) -> str:
        print("[System] 在本地沙盒執行代碼中...")
        result = random.choice(["SUCCESS", "FAIL"])
        return (
            "## Execution Log\n"
            "Running sample task...\n"
            f"[RESULT: {result}]\n"
        )


def _parse_decision(markdown_text: str) -> str | None:
    match = DECISION_PATTERN.search(markdown_text)
    if not match:
        return None
    return match.group(1).upper()


def _parse_result(markdown_text: str) -> str | None:
    match = RESULT_PATTERN.search(markdown_text)
    if not match:
        return None
    return match.group(1).upper()


def _ensure_graphviz_dot_on_path() -> None:
    if shutil.which("dot"):
        return
    graphviz_bin = r"C:\Program Files\Graphviz\bin"
    dot_path = os.path.join(graphviz_bin, "dot.exe")
    if os.path.exists(dot_path):
        os.environ["PATH"] = f"{graphviz_bin};{os.environ.get('PATH', '')}"


def run_workflow(max_steps: int = 20) -> AIWorkflowController:
    _ensure_graphviz_dot_on_path()
    controller = AIWorkflowController()
    controller.get_graph().draw("workflow_graph.png", prog="dot")
    print("[Graph] 已輸出 workflow_graph.png")

    controller.start_project()

    steps = 0
    while controller.state != "idle" and steps < max_steps:
        steps += 1
        if controller.state == "gemini_thinking":
            markdown = controller.mock_call_gemini_cli()
            print(markdown)
            controller.submit_to_review()
            continue

        if controller.state == "chatgpt_reviewing":
            markdown = controller.mock_call_chatgpt_cli()
            print(markdown)
            decision = _parse_decision(markdown)
            if decision == "PASS":
                controller.review_pass()
            elif decision == "REJECT":
                controller.review_fail()
            elif decision == "NEEDS_HUMAN":
                controller.needs_human()
            else:
                raise ValueError(f"無法解析審查決策標籤: {markdown}")
            continue

        if controller.state == "code_executing":
            markdown = controller.mock_call_codex_executor()
            print(markdown)
            result = _parse_result(markdown)
            if result == "SUCCESS":
                controller.execution_success()
            elif result == "FAIL":
                controller.execution_fail()
            else:
                raise ValueError(f"無法解析執行結果標籤: {markdown}")
            continue

        if controller.state == "human_decision":
            print("[Human Decision] 流程暫停，等待人工介入，測試結束。")
            break

    if steps >= max_steps and controller.state != "idle":
        print(f"[Warning] 已達最大步數 {max_steps}，提前結束於狀態: {controller.state}")

    return controller


if __name__ == "__main__":
    run_workflow()
