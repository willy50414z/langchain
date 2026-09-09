# LangChain vs LangGraph：重點與決策條件

## 1) 先講結論

- **LangChain**：適合快速組裝 LLM 能力與中小型流程（輕量、快上手）。
- **LangGraph**：適合管理複雜工作流（多分支、回圈、HITL、恢復、治理）。
- 實務常見做法是：**LangChain 做元件，LangGraph 做流程編排**。

---

## 2) 核心差異（對照）

### 流程形狀
- **LangChain**：多為線性/半線性 pipeline（A -> B -> C）。
- **LangGraph**：狀態機式 graph（可分支、回圈、再入）。

### 狀態管理
- **LangChain**：通常由你在函式中自行管理 state/dict。
- **LangGraph**：state 是一等公民，節點之間明確讀寫與傳遞。

### 控制流能力
- **LangChain**：if/else + while 自行控制，簡單流程很快。
- **LangGraph**：條件邊、節點回圈、路由決策更結構化。

### HITL（Human-in-the-loop）
- **LangChain**：可做，但通常要自己做「暫停/保存/恢復」機制。
- **LangGraph**：原生支援 interrupt/resume（搭配 Studio 更直觀）。

### 可觀測性與治理
- **LangChain**：可追蹤，但多靠你自行加 log/trace。
- **LangGraph**：對流程路徑、節點狀態、人工審核更友善。

### 開發成本
- **LangChain**：前期成本低，PoC 速度快。
- **LangGraph**：前期設計成本較高，但複雜流程可維護性更好。

---

## 3) 決策條件（你可以用這份 checklist）

### 優先用 LangChain（符合多數項）
- 流程是固定 2~5 步驟
- 幾乎沒有回圈與複雜分支
- 不需要 HITL
- 不需要中斷恢復（checkpoint/resume）
- 目前目標是快速驗證需求（PoC）

### 優先用 LangGraph（符合任一關鍵項就可考慮）
- 需要 **HITL**（人工審核後繼續）
- 有 **多分支 + 回圈**（retry/fallback/re-plan）
- 需要 **中斷恢復**（長任務、跨時間執行）
- 需要清楚的流程治理與審計（為何走到某路徑）
- 流程會持續擴張（節點數量會明顯增加）

---

## 4) 你專案中的對照

- `phase2_sample_app.py`
  - 主軸是 RAG 與 LCEL 組裝，偏 LangChain 思維。
- `phase3_langchain_app.py`
  - 不用 LangGraph，也能做 planner/retrieve/tool/evaluator/retry/finalize。
  - 但控制流與狀態更新要自己維護。
- `phase3_sample_app.py`
  - LangGraph-first，展示流程圖式編排（分支/回圈）。
- `phase3_langgraph_hitl.py`
  - 專注 HITL，搭配 `langgraph dev` / Studio 可中斷與人工介入。

---

## 5) 一句話選型法（推薦）

- **先用 LangChain 起步**（快）
- 當流程出現「HITL / 複雜分支回圈 / 恢復需求 / 治理需求」時，**升級到 LangGraph**

這樣能同時兼顧交付速度與長期可維護性。

