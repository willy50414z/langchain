from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from com.willy.langchain.svc.langchain_svc import LangChainService

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = None
    START = None
    StateGraph = None


class DemoHashEmbeddings(Embeddings):
    """教學版 embedding，避免學習階段受外部 API 影響。"""

    def __init__(self, dim: int = 48) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text):
            vec[i % self.dim] += (ord(ch) % 97) / 97.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class Phase3SampleApp:
    """
    LangGraph-first 教學範例：
    planner -> retrieve/tool -> evaluate -> (retry or finalize)
    """

    def __init__(self) -> None:
        self.langchain_svc = LangChainService()
        self.embeddings = DemoHashEmbeddings()
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self) -> InMemoryVectorStore:
        docs = [
            Document(
                page_content=(
                    "LangGraph 適合多步驟、可分支、可回圈的 Agent 工作流。"
                    "常見節點包含 planner、tool、retriever、evaluator、finalizer。"
                ),
                metadata={"source": "phase3-note-1"},
            ),
            Document(
                page_content=(
                    "在企業場景，會加入 checkpoint 與 human-in-the-loop，"
                    "確保任務中斷可恢復且高風險操作可人工審核。"
                ),
                metadata={"source": "phase3-note-2"},
            ),
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=240, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        store = InMemoryVectorStore(embedding=self.embeddings)
        store.add_documents(chunks)
        return store

    @staticmethod
    def _safe_eval(expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Num,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.Mod,
                    ast.USub,
                    ast.UAdd,
                    ast.Load,
                ),
            ):
                raise ValueError("只允許基本數學運算")
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def _synthesize(self, question: str, context: str) -> str:
        prompt = PromptTemplate.from_template(
            (
                "你是 LangGraph 教學助理。\n"
                "請根據 context 回答，若資訊不足請明確說明。\n\n"
                "context:\n{context}\n\n"
                "question:\n{question}\n"
            )
        )
        return self.langchain_svc._generate(prompt.format(context=context, question=question))

    def run_langgraph_workflow(self, user_input: str) -> dict[str, Any]:
        if StateGraph is None:
            return {
                "answer": "尚未安裝 langgraph，請先執行: pip install langgraph",
                "path_trace": [],
                "retry_count": 0,
            }

        def planner(state: dict[str, Any]) -> dict[str, Any]:
            text = state["user_input"].strip()
            is_math = bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", text))
            return {
                "route": "tool" if is_math else "retrieve",
                "path_trace": state["path_trace"] + ["planner"],
            }

        def retrieve(state: dict[str, Any]) -> dict[str, Any]:
            docs = self.vector_store.similarity_search(state["user_input"], k=2)
            context = self._format_docs(docs)
            draft = self._synthesize(question=state["user_input"], context=context)
            return {
                "context": context,
                "draft_answer": draft,
                "path_trace": state["path_trace"] + ["retrieve"],
            }

        def tool(state: dict[str, Any]) -> dict[str, Any]:
            result = self._safe_eval(state["user_input"])
            return {
                "draft_answer": f"計算結果: {result}",
                "path_trace": state["path_trace"] + ["tool"],
            }

        def evaluator(state: dict[str, Any]) -> dict[str, Any]:
            draft = state.get("draft_answer", "").strip()
            retry_count = state.get("retry_count", 0)
            # 教學用評估規則：答案太短且仍有重試額度，走 retry
            need_retry = len(draft) < 12 and state["route"] == "retrieve" and retry_count < 1
            return {
                "decision": "retry" if need_retry else "finalize",
                "retry_count": retry_count + 1 if need_retry else retry_count,
                "path_trace": state["path_trace"] + ["evaluator"],
            }

        def rewrite_query(state: dict[str, Any]) -> dict[str, Any]:
            rewritten = f"{state['user_input']}（請提供更具體重點）"
            return {
                "user_input": rewritten,
                "path_trace": state["path_trace"] + ["rewrite_query"],
            }

        def finalize(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "answer": state.get("draft_answer", "無法產生答案"),
                "path_trace": state["path_trace"] + ["finalize"],
            }

        workflow = StateGraph(dict)
        workflow.add_node("planner", planner)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("tool", tool)
        workflow.add_node("evaluator", evaluator)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("finalize", finalize)

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            lambda s: s["route"],
            {"retrieve": "retrieve", "tool": "tool"},
        )
        workflow.add_edge("retrieve", "evaluator")
        workflow.add_edge("tool", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            lambda s: s["decision"],
            {"retry": "rewrite_query", "finalize": "finalize"},
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("finalize", END)

        graph = workflow.compile()
        init_state = {
            "user_input": user_input,
            "route": "",
            "context": "",
            "draft_answer": "",
            "decision": "",
            "retry_count": 0,
            "path_trace": [],
            "answer": "",
        }
        return graph.invoke(init_state)

    def run(self) -> None:
        print("=== Phase 3 / LangGraph-first Workflow ===")

        q1 = "LangGraph 下一階段學習重點是什麼？"
        result1 = self.run_langgraph_workflow(q1)
        print(f"\n[問題] {q1}")
        print(f"[路徑] {' -> '.join(result1['path_trace'])}")
        print(f"[重試次數] {result1['retry_count']}")
        print(f"[答案] {result1['answer']}")

        q2 = "36 / (3 + 3)"
        result2 = self.run_langgraph_workflow(q2)
        print(f"\n[問題] {q2}")
        print(f"[路徑] {' -> '.join(result2['path_trace'])}")
        print(f"[答案] {result2['answer']}")


if __name__ == "__main__":
    app = Phase3SampleApp()
    try:
        app.run()
    except EnvironmentError as exc:
        print(f"[Environment Error] {exc}")
        print("Please set [local_model] base_url / chat_model / llm_model in application.ini (repo root).")
        sys.exit(1)

    # Phase 3 學習內容項目（LangGraph 主軸）：
    # 1) State 設計：定義每個節點共享與更新的狀態欄位（route/context/draft/decision/path_trace）
    # 2) 節點拆分：planner、retrieve、tool、evaluator、rewrite_query、finalize 各司其職
    # 3) 條件分支：planner 依任務型態路由；evaluator 決定 retry 或 finalize
    # 4) 回圈控制：透過 retry_count 限制重試次數，避免無限迴圈
    # 5) 可觀測性：path_trace 記錄實際經過節點，方便除錯與教學展示
