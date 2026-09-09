# LangChain 學習地圖 (Enterprise Edition)

這份學習地圖旨在引導開發者從基礎概念出發，逐步掌握 LangChain 框架，最終能夠構建生產級的 AI Agent 與 RAG 系統。

---

## 🗺️ 學習路線圖概覽
1. **基礎層 (Foundations)**：LLM 原理、Prompt 工程、LCEL 語法。
2. **數據層 (Data & RAG)**：向量資料庫、嵌入模型、檢索優化。
3. **邏輯層 (Chains & Logic)**：複雜工作流、記憶體管理、工具調用。
4. **進階與生產 (Advanced & Production)**：LangGraph、多 Agent 協作、LangSmith 監控。

---

## 🟢 第一階段：基礎基礎 (Foundations)
*目標：掌握與 LLM 互動的核心語法與基礎概念。*

- **LLM 基礎知識**
  - Token 運算與模型限制 (Context Window)
  - 溫度 (Temperature) 與 Top-P 參數調優
- **Prompt Engineering**
  - Few-shot Prompting
  - Chain of Thought (CoT)
  - System Prompt 設計
- **LangChain Expression Language (LCEL)**
  - `|` 運算子與管道鏈接
  - `Runnable` 介面 (invoke, batch, stream)
  - 預處理與解析 (RunnableParallel, RunnableLambda)

## 🟡 第二階段：數據與 RAG (Retrieval Augmented Generation)
*目標：構建基於私有知識庫的問答系統。*

- **數據處理 (Data Ingestion)**
  - Document Loaders (PDF, Web, Notion, Notion)
  - Text Splitters (RecursiveCharacterTextSplitter)
- **向量資料庫 (Vector Stores)**
  - Pinecone, Milvus, Chroma, FAISS
  - 嵌入模型 (Embeddings) 選型
- **RAG 優化技術**
  - 多查詢重寫 (Multi-Query)
  - 重排序 (Reranking)
  - 混合檢索 (Hybrid Search)

## 🟠 第三場：代理與工具 (Agents & Tools)
*目標：讓 AI 具備執行任務與操作外部工具的能力。*

- **工具調用 (Tool Calling)**
  - 定義工具 (Tool Decorators)
  - 函數調用 (Function Calling)
- **Agent 邏輯**
  - ReAct 框架
  - Plan-and-Execute 模式
- **記憶體管理 (Memory)**
  - Window Buffer, Summary Memory
  - 持久化記憶體 (SQL/Redis)

## 🔴 第四階段：進階架構與生產 (Advanced & Production)
*目標：構建複雜的狀態機與可監控的生產環境。*

- **LangGraph (核心推薦)**
  - 狀態機 (State Machines)
  - 循環圖與條件分支
  - 人機互動 (Human-in-the-loop)
- **多代理協作 (Multi-Agent Systems)**
  - 角色分工與協作流
  - 代理間的通訊協議
- **監控與評估 (Ops)**
  - **LangSmith**: 追蹤、調試與評估
  - **RAGAS**: 自動化評估 RAG 性能
  - 提示詞版本管理 (Prompt Management)

---

## 🛠️ 推薦工具棧
- **框架**: LangChain, LangGraph
- **向量庫**: Pinecone (雲端), Chroma (本地)
- **監控**: LangSmith
- **開發環境**: Jupyter Notebook, VS Code
- **模型提供商**: OpenAI, Anthropic, Groq (高速推理)

## 🎯 實踐專案建議
1. **入門**：建立一個能讀取 PDF 並回答問題的簡單 Chatbot。
2. **進階**：開發一個能自動查詢天氣、查閱資料庫並發送郵件的 Agent。
3. **企業級**：使用 LangGraph 構建一個具有多個角色（如：編程助手、審查員、文案專家）的協作工作流。
---
*最後更新日期：2024-05-22*
*版本：v2.0 Enterprise Edition*
