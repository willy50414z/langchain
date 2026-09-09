# 需求規格書：AI 多代理人協作狀態機 PoC (基於 transitions)

## 1. 專案目標
請使用 Python 的 `transitions` 套件（需包含 `GraphMachine` 可視化擴展），實作一個控制多個 AI 代理人協作的狀態機。
本階段為 Proof of Concept (PoC)，無需真實呼叫外部 AI API，只需實作狀態流轉邏輯、解析模擬的 Markdown 標籤，並成功輸出狀態機架構圖。

## 2. 技術堆疊與依賴
- Python 3.9+
- 套件：`transitions[diagrams]` (用於狀態機與繪圖)
- 系統依賴：Graphviz (用於輸出 PNG 圖片)

## 3. 狀態機定義 (State Machine Architecture)

請建立一個名為 `AIWorkflowController` 的類別，並定義以下狀態 (States) 與流轉規則 (Transitions)：

### 3.1 狀態列表 (States)
1. `idle`: 初始待機狀態。
2. `gemini_thinking`: 正在呼叫 Gemini 生成程式碼。
3. `chatgpt_reviewing`: 正在呼叫 ChatGPT 審查程式碼。
4. `human_decision`: 流程暫停，等待人類介入決策。
5. `code_executing`: 正在本地或沙盒環境執行程式碼。

### 3.2 流轉規則 (Transitions)
- `start_project`: `idle` -> `gemini_thinking`
- `submit_to_review`: `gemini_thinking` -> `chatgpt_reviewing`
- `review_pass`: `chatgpt_reviewing` -> `code_executing`
- `review_fail`: `chatgpt_reviewing` -> `gemini_thinking`
- `needs_human`: `chatgpt_reviewing` -> `human_decision`
- `human_override_pass`: `human_decision` -> `code_executing`
- `human_give_new_prompt`: `human_decision` -> `gemini_thinking`
- `execution_success`: `code_executing` -> `idle`
- `execution_fail`: `code_executing` -> `gemini_thinking`

## 4. 介面定義 (Mock Interfaces)

請在 `AIWorkflowController` 中定義以下三個 Mock 方法。這些方法不需要真實連網，只需印出模擬的 Log，並回傳固定的格式字串，用以模擬 AI 輸出的 Markdown 文件。

### 4.1 `mock_call_gemini_cli()`
- **行為**：印出 `[System] 呼叫 Gemini CLI 生成代碼...`
- **回傳**：模擬的 Markdown 字串，包含一段簡單的 Python 虛擬碼。

### 4.2 `mock_call_chatgpt_cli()`
- **行為**：印出 `[System] 呼叫 ChatGPT CLI 進行代碼審查...`
- **回傳**：模擬的 Markdown 字串。請隨機回傳包含結尾標籤 `[DECISION: PASS]` 或 `[DECISION: REJECT]` 或 `[DECISION: NEEDS_HUMAN]` 的字串。

### 4.3 `mock_call_codex_executor()`
- **行為**：印出 `[System] 在本地沙盒執行代碼中...`
- **回傳**：模擬的執行日誌。請隨機回傳包含結尾標籤 `[RESULT: SUCCESS]` 或 `[RESULT: FAIL]` 的字串。

## 5. 執行流程與測試腳本 (Main Execution)

請撰寫一個 `main` 區塊，模擬一次自動化執行流程：

1. 實例化 `AIWorkflowController` 並綁定 `GraphMachine`，設定 `initial='idle'`。
2. 呼叫 `get_graph().draw('workflow_graph.png', prog='dot')` 將狀態機繪製成圖檔儲存。
3. 觸發 `start_project()` 進入 `gemini_thinking`。
4. 撰寫一個簡單的 `while` 迴圈或條件判斷，模擬自動化流程：
   - 當處於 `gemini_thinking`，呼叫 Mock Gemini，然後觸發 `submit_to_review()`。
   - 當處於 `chatgpt_reviewing`，呼叫 Mock ChatGPT，透過簡單的正則表達式 (Regex) 或字串搜尋解析回傳的標籤 (`[DECISION: ...]`)，並觸發對應的流轉方法 (`review_pass()`, `review_fail()` 等)。
   - 當處於 `code_executing`，呼叫 Mock Executor，解析標籤 (`[RESULT: ...]`) 並觸發成功或失敗的方法。
   - 若進入 `human_decision`，印出提示訊息並 `break` 結束測試流程。
5. 在每個狀態轉換後，印出 `[State Change] 當前狀態: {當前狀態名稱}`。

請提供完整的 Python 程式碼，確保可以直接執行並產出圖片。