from __future__ import annotations

import ast
import re
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


class Phase3State(TypedDict):
    user_input: str
    route: str
    context: str
    draft_answer: str
    answer: str
    approved: bool
    path_trace: list[str]


class DemoHashEmbeddings(Embeddings):
    """教學用 embedding，避免 HITL demo 依賴外部 embedding API。"""

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


def _build_store() -> InMemoryVectorStore:
    docs = [
        Document(
            page_content=(
                "LangGraph 的 HITL 常見模式：模型先產生草稿，"
                "人類在關鍵節點審核與修改，再繼續流程。"
            ),
            metadata={"source": "hitl-note-1"},
        ),
        Document(
            page_content=(
                "企業流程中，人工審核節點通常放在最終輸出前，"
                "可避免高風險資訊直接發送。"
            ),
            metadata={"source": "hitl-note-2"},
        ),
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    store = InMemoryVectorStore(embedding=DemoHashEmbeddings())
    store.add_documents(chunks)
    return store


STORE = _build_store()


def _ensure_defaults(state: Phase3State) -> Phase3State:
    """
    Studio 可能只傳入部分欄位，先補預設值，避免 KeyError。
    這樣使用者只填 user_input 也可以啟動流程。
    """
    state.setdefault("user_input", "")
    state.setdefault("route", "")
    state.setdefault("context", "")
    state.setdefault("draft_answer", "")
    state.setdefault("answer", "")
    state.setdefault("approved", False)
    state.setdefault("path_trace", [])
    return state


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


def planner(state: Phase3State) -> Phase3State:
    state = _ensure_defaults(state)
    text = state["user_input"].strip()
    if not text:
        state["route"] = "finalize"
        state["draft_answer"] = "請先輸入問題內容，再執行流程。"
        state["path_trace"] = state["path_trace"] + ["planner"]
        return state
    is_math = bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", text))
    state["route"] = "tool" if is_math else "retrieve"
    state["path_trace"] = state["path_trace"] + ["planner"]
    return state


def retrieve(state: Phase3State) -> Phase3State:
    state = _ensure_defaults(state)
    docs = STORE.similarity_search(state["user_input"], k=2)
    context = "\n\n".join(d.page_content for d in docs)
    state["context"] = context
    state["draft_answer"] = f"根據檢索內容，建議答案草稿：\n{context}"
    state["path_trace"] = state["path_trace"] + ["retrieve"]
    return state


def tool(state: Phase3State) -> Phase3State:
    state = _ensure_defaults(state)
    value = _safe_eval(state["user_input"])
    state["draft_answer"] = f"計算結果草稿: {value}"
    state["path_trace"] = state["path_trace"] + ["tool"]
    return state


def human_review(state: Phase3State) -> Phase3State:
    state = _ensure_defaults(state)
    # 在 Studio 中會先被 interrupt_before 攔下；人工可修改 draft_answer/approved 後再 resume。
    state["path_trace"] = state["path_trace"] + ["human_review"]
    return state


def finalize(state: Phase3State) -> Phase3State:
    state = _ensure_defaults(state)
    state["answer"] = state["draft_answer"]
    state["path_trace"] = state["path_trace"] + ["finalize"]
    return state


workflow = StateGraph(Phase3State)
workflow.add_node("planner", planner)
workflow.add_node("retrieve", retrieve)
workflow.add_node("tool", tool)
workflow.add_node("human_review", human_review)
workflow.add_node("finalize", finalize)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner",
    lambda s: s["route"],
    {"retrieve": "retrieve", "tool": "tool", "finalize": "finalize"},
)
workflow.add_edge("retrieve", "human_review")
workflow.add_edge("tool", "human_review")
workflow.add_edge("human_review", "finalize")
workflow.add_edge("finalize", END)

# 關鍵：在 human_review 前中斷，讓 Studio 可進行 HITL 審核與編修。
graph = workflow.compile(interrupt_before=["human_review"])
