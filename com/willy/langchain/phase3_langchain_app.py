from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from com.willy.langchain.svc.langchain_svc import LangChainService


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


class Phase3LangChainApp:
    """
    LangChain 版本 Phase 3：
    - 使用 LCEL 組 retrieval + synthesis
    - 用 Python while/if 實作分支與重試控制（不使用 LangGraph）
    """

    def __init__(self) -> None:
        self.langchain_svc = LangChainService()
        self.embeddings = DemoHashEmbeddings()
        self.vector_store = self._build_vector_store()
        self.rag_chain = self._build_rag_chain()

    def _build_vector_store(self) -> InMemoryVectorStore:
        docs = [
            Document(
                page_content=(
                    "LangChain 偏向元件組合與流程串接。"
                    "若流程主要是固定步驟，可用 LCEL 保持可讀與可維護。"
                ),
                metadata={"source": "phase3-langchain-note-1"},
            ),
            Document(
                page_content=(
                    "當流程需要大量分支、回圈、人工介入與恢復時，"
                    "通常會升級到 LangGraph。"
                ),
                metadata={"source": "phase3-langchain-note-2"},
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
                "你是 LangChain 教學助理。\n"
                "請根據 context 回答，若資訊不足請明確說明。\n\n"
                "context:\n{context}\n\n"
                "question:\n{question}\n"
            )
        )
        return self.langchain_svc._generate(prompt.format(context=context, question=question))

    def _build_rag_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        return (
            {
                "context": retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: self._synthesize(question=x["question"], context=x["context"]))
        )

    def run_langchain_workflow(self, user_input: str) -> dict[str, Any]:
        state: dict[str, Any] = {
            "user_input": user_input,
            "route": "",
            "context": "",
            "draft_answer": "",
            "decision": "",
            "retry_count": 0,
            "path_trace": [],
            "answer": "",
        }

        # planner
        text = state["user_input"].strip()
        is_math = bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", text))
        state["route"] = "tool" if is_math else "retrieve"
        state["path_trace"].append("planner")

        # 用 while 展示非圖式流程如何做 retry 控制
        while True:
            if state["route"] == "tool":
                state["draft_answer"] = f"計算結果: {self._safe_eval(state['user_input'])}"
                state["path_trace"].append("tool")
            else:
                docs = self.vector_store.similarity_search(state["user_input"], k=2)
                state["context"] = self._format_docs(docs)
                state["draft_answer"] = self.rag_chain.invoke(state["user_input"])
                state["path_trace"].append("retrieve")

            # evaluator
            draft = state.get("draft_answer", "").strip()
            need_retry = len(draft) < 12 and state["route"] == "retrieve" and state["retry_count"] < 1
            state["decision"] = "retry" if need_retry else "finalize"
            state["path_trace"].append("evaluator")

            if state["decision"] == "retry":
                state["retry_count"] += 1
                state["user_input"] = f"{state['user_input']}（請提供更具體重點）"
                state["path_trace"].append("rewrite_query")
                continue
            break

        # finalize
        state["answer"] = state.get("draft_answer", "無法產生答案")
        state["path_trace"].append("finalize")
        return state

    def run(self) -> None:
        print("=== Phase 3 / LangChain Workflow (No LangGraph) ===")

        q1 = "LangChain 與 LangGraph 的選型原則是什麼？"
        result1 = self.run_langchain_workflow(q1)
        print(f"\n[問題] {q1}")
        print(f"[路徑] {' -> '.join(result1['path_trace'])}")
        print(f"[重試次數] {result1['retry_count']}")
        print(f"[答案] {result1['answer']}")

        q2 = "36 / (3 + 3)"
        result2 = self.run_langchain_workflow(q2)
        print(f"\n[問題] {q2}")
        print(f"[路徑] {' -> '.join(result2['path_trace'])}")
        print(f"[答案] {result2['answer']}")


if __name__ == "__main__":
    app = Phase3LangChainApp()
    try:
        app.run()
    except EnvironmentError as exc:
        print(f"[Environment Error] {exc}")
        print("Please set [local_model] base_url / chat_model / llm_model in application.ini (repo root).")
        sys.exit(1)

    # 對照重點：
    # 1) 這版不用 LangGraph，也能做 planner/retrieve/tool/evaluator/retry/finalize
    # 2) 但流程控制、狀態更新、迴圈結束條件都要自行維護
    # 3) 當流程變複雜（多分支、HITL、checkpoint）時，維護成本會快速上升
    #
    # 跟 phase3_sample_app.py（LangGraph 版）的逐點對照：
    # A. 流程定義方式
    #    - phase3_sample_app.py: 用 StateGraph.add_node/add_edge 宣告式定義流程圖
    #    - phase3_langchain_app.py: 用 while + if/else 以命令式手動控制流程
    #
    # B. 路由與分支
    #    - LangGraph 版: add_conditional_edges() 將條件與路徑映射清楚分離
    #    - LangChain 版: 在主流程中直接寫判斷，邏輯集中但容易越寫越長
    #
    # C. 重試/回圈
    #    - LangGraph 版: evaluator 決策後回到 rewrite_query 再進 retrieve（圖上可見）
    #    - LangChain 版: 以 while + continue 實作，需自行防止無限迴圈
    #
    # D. 狀態管理
    #    - LangGraph 版: 以 state 在節點間傳遞，每節點只處理局部更新
    #    - LangChain 版: 由單一函式集中修改 dict，彈性高但責任邊界較模糊
    #
    # E. 可觀測性
    #    - LangGraph 版: 天然對接 Studio，可直接看到節點移動與中斷點
    #    - LangChain 版: 主要靠 path_trace/print 追蹤，缺少圖形化執行檢視
    #
    # F. HITL 能力
    #    - LangGraph 版: 可加 interrupt_before/after，在節點前暫停等待人工介入
    #    - LangChain 版: 需自行實作等待、保存狀態、resume 機制
    #
    # G. 適用場景
    #    - LangGraph 版: 多分支、多角色、需審核與恢復的企業流程
    #    - LangChain 版: 線性或中等複雜流程，追求快速實作與低框架負擔
