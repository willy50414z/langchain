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

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
except ImportError:
    PyPDFLoader = None
    TextLoader = None
    WebBaseLoader = None

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = None
    START = None
    StateGraph = None


class DemoHashEmbeddings(Embeddings):
    """教學用 Embedding：不依賴外部 API，方便先理解向量化流程。"""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for index, char in enumerate(text):
            vec[index % self.dim] += (ord(char) % 97) / 97.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class Phase2SampleApp:
    """RAG + 流程編排學習範例（LCEL + LangGraph）。"""

    def __init__(self) -> None:
        self.langchain_svc = LangChainService()
        self.learning_data_dir = Path(__file__).resolve().parent / "learning_data"
        self.learning_data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = DemoHashEmbeddings(dim=48)

    def _prepare_local_txt_file(self) -> Path:
        """建立本機 TXT 範例文件，讓你先從穩定資料源學習 RAG。"""
        txt_path = self.learning_data_dir / "rag_notes.txt"
        if not txt_path.exists():
            txt_path.write_text(
                (
                    "RAG 核心流程: 文件載入 -> 切塊 -> 向量化 -> 儲存 -> 檢索 -> LLM 綜合。\n"
                    "向量檢索可讓模型查到最新知識，而非只依賴預訓練記憶。\n"
                    "企業場景常見做法是先取回 top-k 文件，再把 context 與問題一起交給 LLM。"
                ),
                encoding="utf-8",
            )
        return txt_path

    def _load_documents(self) -> list[Document]:
        """
        混合資料源載入示範：
        1) TXT：最穩定、最容易重現
        2) Web：展示外部內容接入
        3) PDF：若你提供 sample.pdf 且安裝對應套件即可啟用
        """
        docs: list[Document] = []
        txt_path = self._prepare_local_txt_file()

        if TextLoader is not None:
            docs.extend(TextLoader(str(txt_path), encoding="utf-8").load())
        else:
            docs.append(Document(page_content=txt_path.read_text(encoding="utf-8"), metadata={"source": "txt-fallback"}))

        if WebBaseLoader is not None:
            docs.extend(
                WebBaseLoader("https://python.langchain.com/docs/concepts/rag/").load()[:1]
            )
        else:
            docs.append(
                Document(
                    page_content="Web loader 未安裝，因此以本地 fallback 文字替代。",
                    metadata={"source": "web-fallback"},
                )
            )

        sample_pdf = self.learning_data_dir / "sample.pdf"
        if sample_pdf.exists() and PyPDFLoader is not None:
            docs.extend(PyPDFLoader(str(sample_pdf)).load())

        return docs

    @staticmethod
    def _format_docs(documents: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    def _build_vector_store(self) -> InMemoryVectorStore:
        """把文件切塊後存入向量資料庫，成為可檢索知識庫。"""
        raw_docs = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=80)
        chunks = splitter.split_documents(raw_docs)

        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(chunks)
        return vector_store

    def _llm_synthesize(self, question: str, context: str) -> str:
        """把檢索到的 context 與問題交給 LLM 做綜合回答。"""
        prompt = PromptTemplate.from_template(
            (
                "你是 RAG 教學助理，請僅根據 context 回答。\n"
                "若 context 沒有答案，明確說明缺少資訊。\n\n"
                "context:\n{context}\n\n"
                "question:\n{question}\n\n"
                "請用繁體中文條列重點回答。"
            )
        )
        rendered_prompt = prompt.format(context=context, question=question)
        return self.langchain_svc._generate(rendered_prompt)

    def run_lcel_rag_demo(self, question: str) -> str:
        """
        LCEL 版本 RAG：
        Chunking -> VectorStore -> Retrieval -> LLM Synthesis
        """
        vector_store = self._build_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            {
                "context": retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: self._llm_synthesize(question=x["question"], context=x["context"]))
        )
        return rag_chain.invoke(question)

    @staticmethod
    def _safe_eval(expression: str) -> float:
        """簡易安全計算器：僅允許數字與四則運算。"""
        tree = ast.parse(expression, mode="eval")
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
                raise ValueError("只允許數學運算式")
        return float(eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {}))

    def run_agent_like_router(self, user_input: str) -> str:
        """
        Agent 概念（不依賴 LangGraph）：
        根據輸入內容動態決策「呼叫計算器」或「走 RAG 檢索」。
        """
        math_expr_match = re.search(r"[0-9\.\+\-\*\/\(\)\s]{3,}", user_input)
        if math_expr_match and re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", user_input.strip()):
            value = self._safe_eval(user_input.strip())
            return f"偵測到數學任務，計算結果: {value}"
        return self.run_lcel_rag_demo(user_input)

    def run_langgraph_demo(self, user_input: str) -> str:
        """
        LangGraph 版本流程編排：
        route -> (calculator 或 rag) -> END
        """
        if StateGraph is None:
            return "尚未安裝 langgraph，請先 `pip install langgraph`。"

        def route(state: dict[str, Any]) -> dict[str, Any]:
            text = state["user_input"].strip()
            is_math = bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", text))
            return {"route": "calculator" if is_math else "rag"}

        def call_calculator(state: dict[str, Any]) -> dict[str, Any]:
            return {"answer": f"[LangGraph 計算器] {self._safe_eval(state['user_input'])}"}

        def call_rag(state: dict[str, Any]) -> dict[str, Any]:
            return {"answer": self.run_lcel_rag_demo(state["user_input"])}

        workflow = StateGraph(dict)
        workflow.add_node("route", route)
        workflow.add_node("calculator", call_calculator)
        workflow.add_node("rag", call_rag)
        workflow.add_edge(START, "route")
        workflow.add_conditional_edges("route", lambda s: s["route"], {"calculator": "calculator", "rag": "rag"})
        workflow.add_edge("calculator", END)
        workflow.add_edge("rag", END)
        graph = workflow.compile()

        result = graph.invoke({"user_input": user_input})
        return result["answer"]

    def run(self) -> None:
        """主程式：依序展示 RAG 與流程編排能力。"""
        print("=== Learning Sample / Loading + Embedding + RAG ===")
        answer = self.run_lcel_rag_demo("RAG 的企業實作流程通常如何設計？")
        print(answer)

        print("\n=== Learning Sample / Agent-like Router ===")
        print(self.run_agent_like_router("18 * (7 + 5)"))
        print(self.run_agent_like_router("為什麼 RAG 要做切塊與向量儲存？"))

        print("\n=== Learning Sample / LangGraph ===")
        print(self.run_langgraph_demo("3 + 9 / 3"))
        print(self.run_langgraph_demo("請說明檢索增強生成的關鍵步驟"))


if __name__ == "__main__":
    app = Phase2SampleApp()
    try:
        app.run()
    except EnvironmentError as exc:
        print(f"[Environment Error] {exc}")
        print("Please set [local_model] base_url / chat_model / llm_model in application.ini (repo root).")
        sys.exit(1)

    # Phase 2 學習內容項目與描述（含可替換選項與特點）：
    # 1. 文件載入與嵌入 (Loading & Embedding)
    #   1.1 Document Loaders（資料來源可替換）
    #       - TextLoader:
    #         特點: 最穩定、最快上手、最容易除錯
    #         適用: 教學驗證、內部純文字知識庫
    #       - WebBaseLoader:
    #         特點: 可讀取最新網頁內容，適合動態知識
    #         適用: 文件網站、公告、FAQ
    #       - PyPDFLoader:
    #         特點: 可直接處理企業常見 PDF 文件
    #         適用: 合約、手冊、報告（建議搭配 OCR/清洗流程）
    #
    #   1.2 Embedding（向量模型可替換）
    #       - DemoHashEmbeddings（本範例）:
    #         特點: 零外部依賴、成本低、可離線學習流程
    #         侷限: 非語意模型，檢索品質有限
    #       - OpenAI Embeddings:
    #         特點: 語意品質高、生態成熟、整合容易
    #         代價: 需雲端 API 與費用
    #       - Gemini Embeddings:
    #         特點: 與 Gemini 生態整合佳，多語言能力強
    #         代價: 需雲端 API 與費用
    #       - BGE / E5 / Instructor 等開源 Embeddings:
    #         特點: 可地端部署、資料不出網、可客製化微調
    #         代價: 需自行維運 GPU/模型服務
    #
    # 2. 建構檢索增強生成 (RAG) 流程
    #   2.1 Chunking（切塊策略可替換）
    #       - RecursiveCharacterTextSplitter（本範例）:
    #         特點: 通用、穩定、適合多數文件
    #       - Token-based splitter:
    #         特點: 更貼近模型 token 限制，長文控制更精準
    #       - Markdown/HTML 結構切塊:
    #         特點: 保留章節語意，檢索結果更可讀
    #
    #   2.2 Vector Store（向量庫可替換）
    #       - InMemoryVectorStore（本範例）:
    #         特點: 無需部署、開發最快
    #         侷限: 不持久化，重啟即失
    #       - FAISS:
    #         特點: 高效近似搜尋、地端部署常見
    #         適用: 單機或批次建索引場景
    #       - Chroma:
    #         特點: 開發體驗佳、容易本地持久化
    #         適用: 中小型專案 PoC
    #       - Milvus / Weaviate / Pinecone:
    #         特點: 分散式擴展、雲端服務能力強
    #         適用: 企業級規模與高併發檢索
    #
    #   2.3 Retrieval（檢索策略可替換）
    #       - Top-k similarity（本範例）:
    #         特點: 實作最簡單、延遲低
    #       - MMR（多樣性檢索）:
    #         特點: 降低重複內容，提升資訊覆蓋率
    #       - Hybrid Search（向量 + 關鍵字）:
    #         特點: 同時兼顧語意與精準詞匹配
    #       - Reranker（二階重排）:
    #         特點: 提升最終相關性，尤其在長文件場景
    #
    #   2.4 Synthesis（答案生成可替換）
    #       - 單次 Prompt Synthesis（本範例）:
    #         特點: 流程直觀、容易教學
    #       - Map-Reduce / Refine:
    #         特點: 大量 context 時更穩定，能分段彙整
    #       - 引用來源 + 信心標記:
    #         特點: 提升可追溯性與企業審計能力
    #
    #   2.5 LCEL Chain（流程組裝可替換）
    #       - Runnable 組合（本範例）:
    #         特點: 宣告式、可測試、可重用
    #       - 傳統 imperative 函式串接:
    #         特點: 入門直覺，但擴展與追蹤較弱
    #
    # 3. 流程編排與 Agent 概念
    #   3.1 Agent-like Router（路由決策可替換）
    #       - 規則式路由（本範例）:
    #         特點: 可預測、成本低、容易控管
    #       - LLM Router:
    #         特點: 彈性高，可根據語意選工具
    #         代價: 成本與不確定性較高
    #
    #   3.2 LangGraph（圖式工作流可替換）
    #       - StateGraph 條件分支（本範例）:
    #         特點: 清楚描述多步驟節點與路徑
    #       - 增加 Human-in-the-loop:
    #         特點: 關鍵節點可人工審核，降低風險
    #       - 增加 Memory/Checkpoint:
    #         特點: 支援長流程恢復、可觀測與除錯
