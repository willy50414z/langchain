from langgraph import phase2_wf, phase3_wf

config = {"configurable":{"thread_id":"session1"}}
req = "output langchain learning map to md"

graph = phase3_wf.build()

# 逐個node執行並return after node state，直到END | HILT
for event in graph.stream({"requirement":req}, config):
    # event = 每個node執行後的State
    # {'plan': {
    #     'plan': "這是一個專業顧問視角下的 **LangChain 學習地圖 (Learning Map)**。\n\n由於 LangChain 是一個快速發展、概念複雜的框架，我將其拆解為 **四個階段 (Phases)**，確保學習路徑是循序漸進、由淺入深，從基礎概念到實際應用，最後達到系統級的掌握。\n\n---\n\n# 🗺️ LangChain 學習地圖 (Learning Map)\n\n**目標受眾：** 具備 Python 基礎，對大型語言模型 (LLMs) 有基本了解的開發者或數據科學家。\n**學習時長預估：** 根據投入時間不同，可從 4 週到 3 個月達到熟練應用。\n**核心理念：** LangChain 的核心價值是將 LLM 從一個「單次 API 呼叫」提升為一個「多步驟、有記憶、能使用工具的應用系統」。\n\n---\n\n## 🚀 階段 0：基礎準備與預備知識 (Prerequisites & Setup)\n\n在開始學習 LangChain 之前，必須確保以下基礎知識到位。\n\n### 📚 學習目標\n*   熟悉 Python 語法，特別是類別 (Classes) 和函式 (Functions) 的使用。\n*   理解 API 呼叫的流程 (Request/Response)。\n*   了解 LLM 的基本概念 (Tokens, Context Window, Prompt Engineering)。\n\n### 🛠️ 關鍵任務\n1.  **環境設置：** 設置 Python 環境，安裝必要的函式庫 (`pip install langchain openai ...`)。\n2.  **API Key 管理：** 學習如何安全地管理和使用 LLM 提供商的 API Key (例如：使用環境變數)。\n3.  **基礎 LLM 呼叫：** 能夠使用 OpenAI 或其他 LLM SDK，進行最簡單的文本生成呼叫。\n\n---\n\n## 🧱 階段 1：核心概念與基礎組件 (Fundamentals & Core Components)\n\n此階段是理解 LangChain 框架的骨架，掌握其最基本的「積木」概念。\n\n### 🎯 學習目標\n*   理解 LangChain 的設計哲學：將複雜的 LLM 應用拆解為可管理的模組。\n*   掌握 LangChain 的三大核心組件：`Prompt`、`Chain`、`LLM`。\n\n### 💡 關鍵概念與實作\n| 組件 (Component) | 概念理解 (What is it?) | 實作任務 (How to use it?) |\n| :--- | :--- | :--- |\n| **Prompt Templates** | 如何結構化地輸入指令，確保輸入的變數是可控的。 | 實作一個接受用戶輸入和系統指令的 Prompt 模板。 |\n| **LLMs / Chat Models** | 了解不同模型類型（文本 vs. 聊天）的差異和呼叫方式。 | 進行簡單的問答 (Q&A) 流程，並觀察不同模型輸出的差異。 |\n| **Chains** | 這是 LangChain 的核心。理解「步驟序列化」的概念。 | 實作一個簡單的 **兩步驟鏈 (Two-step Chain)**：步驟一（摘要） $\\rightarrow$ 步驟二（翻譯）。 |\n| **Output Parsers** | 如何將 LLM 輸出的非結構化文本，轉換成程式碼可讀的結構化格式 (如 JSON)。 | 讓 LLM 根據特定格式（如 JSON Schema）輸出結果，並用 Parser 驗證。 |\n\n---\n\n## 🧠 階段 2：進階應用與知識擴充 (Advanced Topics & Knowledge Augmentation)\n\n此階段是將 LangChain 的能力從單純的文本生成，提升到「能讀書、能思考」的層次。\n\n### 🎯 學習目標\n*   掌握 RAG (Retrieval-Augmented Generation) 的完整流程。\n*   理解「記憶」和「工具使用」的概念。\n*   能夠處理外部數據源的整合。\n\n### 📚 關鍵概念與實作\n| 組件 (Component) | 概念理解 (What is it?) | 實作任務 (How to use it?) |\n| :--- | :--- | :--- |\n| **Document Loaders** | 如何從各種來源（PDF, 網頁, 資料庫）載入文件。 | 載入一個本地 PDF 文件，並將其內容切塊 (Chunking)。 |\n| **Embeddings & Vector Stores** | 了解「向量化」的原理，以及如何使用向量資料庫 (如 Chroma, Pinecone) 進行相似度檢索。 | 建立一個簡單的知識庫：將文件切塊 $\\rightarrow$ 向量化 $\\rightarrow$ 存入 Vector Store。 |\n| **Retrieval Chain (RAG)** | 這是最常見的應用。流程：提問 $\\rightarrow$ 檢索相關文件 $\\rightarrow$ 將文件作為上下文 $\\rightarrow$ 讓 LLM 回答。 | **實作一個基於本地文件的問答機器人 (Chatbot)**。這是本階段的必修課。 |\n| **Memory** | 如何讓應用程式「記住」過去的對話歷史。 | 實作一個帶有對話記憶的聊天機器人，確保它能參考前幾輪的對話內容。 |\n| **Agents & Tools** | 這是 LangChain 最強大的部分。Agent 讓 LLM 具備「決策能力」，能決定使用哪些工具。 | 讓 Agent 具備「搜尋工具 (Search Tool)」和「計算器工具 (Calculator Tool)」，並讓它自主決定使用哪個工具來回答問題。 |\n\n---\n\n## 🌐 階段 3：系統化、優化與部署 (Systemization, Optimization & Deployment)\n\n達到熟練應用級別。重點從「寫出功能」轉移到「讓功能穩定、高效、可維護」。\n\n### 🎯 學習目標\n*   掌握 LangChain 生態系統的進階工具。\n*   了解應用程式的監控、優化和部署流程。\n\n### 🛠️ 關鍵任務\n1.  **LangSmith (必學)：** 學習使用 LangSmith 進行 **Tracing (追蹤)** 和 **Debugging (除錯)**。這對於複雜的 Agent 流程至關重要。\n2.  **Prompt Optimization：** 學習 Few-Shot Learning 和 Chain-of-Thought (CoT) 等進階 Prompt 技術，提升 LLM 的推理能力。\n3.  **LangServe/FastAPI 部署：** 將開發好的 LangChain 應用，封裝成一個可供外部調用的 API 服務。\n4.  **多模態整合 (Optional)：** 探索如何將圖像、音訊等非文本數據，整合到 LangChain 的流程中。\n\n---\n\n## 🏆 總結與學習路徑建議 (Consultant's Summary)\n\n| 階段 | 掌握的技能 | 關鍵產出 (Deliverable) | 學習重點 |\n| :--- | :--- | :--- | :--- |\n| **0. 基礎** | Python, API 呼叫, LLM 概念 | 簡單的文本生成腳本 | 建立信心，理解輸入/輸出的流程。 |\n| **1. 基礎組件** | Prompt, Chain, Parser | 兩步驟的串聯應用 (e.g., 摘要 $\\rightarrow$ 翻譯) | 掌握「流程控制」和「結構化輸出」。 |\n| **2. 進階應用** | RAG, Memory, Agents, Tools | **完整的 RAG 聊天機器人** | 掌握「外部知識獲取」和「自主決策」。 |\n| **3. 系統化** | LangSmith, LangServe, CoT | **可部署的、帶有監控的 API 服務** | 掌握「工程化」和「穩定性」。 |\n\n### 💡 學習建議 (Action Items)\n\n1.  **不要只看教學，一定要實作：** 每個階段的知識點，都必須用自己的代碼跑一遍。\n2.  **從「問題」出發：** 不要從「LangChain 的組件」出發，而是從「我想解決什麼問題？」出發，然後反推需要哪些組件。\n3.  **優先順序：** 建議先專注於 **RAG (階段 2)**，這是目前企業應用中最實用、最容易看到成效的領域。\n\n---\n***Disclaimer:*** *此學習地圖是一個高層次的指引，實際學習時，請務必參考 LangChain 官方文件 (Documentation) 獲取最新的 API 參數和最佳實踐。*"}}
    # {'review_plan': {
    #     'plan': '這是一份**極為專業且結構完整**的學習地圖。從一個顧問的角度來看，您已經掌握了 LangChain 整個生態系統的骨架，並且成功地將複雜的技術學習路徑，轉化為一個可量化的、階段性的學習藍圖。\n\n**總體評價：A+ (Excellent)**\n\n您成功地將一個「技術棧」的學習，轉化為一個「工程化流程」的學習。這已經遠超一般初學者或單純的教學文件。\n\n---\n\n## 🔍 嚴格審查與優化建議 (Expert Review & Critique)\n\n雖然內容極為完善，但作為一位嚴格的計畫審查專家，我必須指出幾個**「過度設計」**和**「學習負荷過重」**的風險點。\n\n### ⚠️ 1. 最大的風險：學習範圍過廣 (Scope Creep)\n您涵蓋了從基礎組件到部署、從 RAG 到 Agents，幾乎囊括了 LangChain 的所有核心功能。這在規劃上是完美的，但在執行上，這會給學習者帶來巨大的**「認知超載 (Cognitive Overload)」**。學習者可能會在每個組件上花費過多時間，導致無法完成任何一個完整的專案。\n\n**【修正方向】**：必須為每個階段設定一個**「最小可行專案 (Minimum Viable Project, MVP)」**，將學習的重點從「學會組件」轉移到「完成專案」。\n\n### ⚠️ 2. 流程的銜接點需要加強 (Transition Gap)\n從 **階段 1 (基礎組件)** 躍升到 **階段 2 (RAG/Agents)** 的跨度太大了。RAG 涉及了「文件處理」、「向量化」、「資料庫」等多個獨立的技術棧。如果學習者沒有在階段 1 建立起「數據輸入 $\\rightarrow$ 處理 $\\rightarrow$ 輸出」的完整心智模型，直接進入 RAG 會感到非常吃力。\n\n**【修正方向】**：在階段 1 和階段 2 之間，需要一個**「數據處理橋樑 (Data Processing Bridge)」**的過渡環節。\n\n### ⚠️ 3. 優先級的指導不足 (Lack of Priority)\n您將 LangSmith、FastAPI、多模態整合等都列為重要任務。對於一個初學者來說，這些是**「工程化優化」**的技能，而不是**「核心功能實現」**的技能。如果將它們放在同等重要性的位置，會讓學習者混淆學習的優先順序。\n\n**【修正方向】**：將學習路徑劃分為三個明確的階段：**功能實現 $\\rightarrow$ 穩定優化 $\\rightarrow$ 商業化部署**。\n\n---\n\n## ✨ 優化後的執行計畫：專案驅動的 LangChain 學習路徑 (Optimized Plan)\n\n我將原有的學習地圖，調整為一個更具**「專案導向 (Project-Driven)」**和**「遞進式難度 (Progressive Difficulty)」**的路線圖。\n\n**核心原則：** 每個階段的目標不是「學會組件」，而是「完成一個能展示成果的 MVP」。\n\n### 🚀 階段 0：基礎奠基 (Foundation & Setup)\n*   **目標：** 建立 LLM 應用程式的最小心智模型。\n*   **關鍵產出 (MVP)：** 能夠運行一個接受用戶輸入，並返回結構化文本的簡單腳本。\n*   **重點調整：** 保持不變。重點放在環境和 API Key 管理。\n\n### 🧱 階段 1：流程控制與結構化輸出 (The Core Flow)\n*   **目標：** 掌握如何將單次 API 呼叫，變成一個可控的、多步驟的流程。\n*   **關鍵概念：** Prompt Templates $\\rightarrow$ Chain $\\rightarrow$ Output Parsers。\n*   **💡 專案實作 (MVP)：** **「數據轉換器 (Data Transformer)」**\n    *   **流程：** 接收一段非結構化的文字（例如：一段會議記錄）。\n    *   **步驟 1 (Chain)：** 使用 LLM 進行摘要。\n    *   **步驟 2 (Parser)：** 使用 Output Parser 將摘要結果，強制轉換成一個包含 `[日期]`, `[關鍵人物]`, `[行動項目]` 的 JSON 格式。\n    *   **學習重點：** 強調 **「輸入 $\\rightarrow$ 處理 $\\rightarrow$ 結構化輸出」** 的完整流程控制能力。\n\n### 🧠 階段 2：知識擴充與自主決策 (The Intelligence Layer)\n*   **目標：** 讓應用程式具備「讀書能力」和「思考能力」。這是從「聊天機器人」到「企業知識助手」的關鍵跨越。\n*   **關鍵概念：** RAG (核心) $\\rightarrow$ Memory $\\rightarrow$ Agents/Tools。\n*   **💡 專案實作 (MVP)：** **「企業知識問答系統 (RAG Chatbot)」**\n    *   **流程：** 建立一個基於本地文件（PDF/DOCX）的問答機器人。\n    *   **步驟 1 (Data Loader/Chunking)：** 載入文件並切塊。\n    *   **步驟 2 (Vector Store)：** 建立向量資料庫。\n    *   **步驟 3 (RAG Chain)：** 實現檢索 $\\rightarrow$ 上下文注入 $\\rightarrow$ 回答的完整流程。\n    *   **進階升級 (Agent)：** 在 RAG 系統基礎上，增加一個「計算器工具」或「天氣查询工具」，讓 Agent 能夠根據用戶需求，決定是否需要調用外部工具，從而實現更複雜的任務執行。\n\n### 🚀 階段 3：生產化與優化 (Production Readiness)\n*   **目標：** 將原型系統轉化為穩定、可擴展的產品。\n*   **核心技能：**\n    1.  **狀態管理 (State Management):** 如何在多輪對話中保持上下文的連續性。\n    2.  **錯誤處理 (Error Handling):** 處理外部 API 失敗、用戶輸入無效等場景。\n    3.  **部署與監控 (Deployment & Monitoring):** 將應用部署到雲端（如 Streamlit Cloud, AWS Lambda），並設置監控日誌。\n*   **學習重點：** 關注框架層面的最佳實踐，而不是單純的組件組合。\n\n---\n### 總結對比表 (Actionable Roadmap)\n\n| 階段 | 核心目標 | 關鍵技術/概念 | 衡量標準 (KPI) |\n| :--- | :--- | :--- | :--- |\n| **階段 1** | 掌握基礎流程 | Prompt Engineering, 基本 API 調用, 簡單的輸入/輸出處理。 | 成功構建一個能回答單一問題的聊天機器人。 |\n| **階段 2** | 實現複雜邏輯 | **RAG (Retrieval-Augmented Generation)**, Agent 框架 (Tool Calling), 狀態管理。 | 成功構建一個能根據外部文件回答複雜、多步驟問題的系統。 |\n| **階段 3** | 產品化與穩定性 | 錯誤處理, 部署流程, 性能優化, 成本控制。 | 系統能夠穩定運行 24/7，並能處理真實世界的邊界條件。 |'}}
    print(event)

#進入HILT
# 2. 檢視當前狀態（人類審核階段）🔍
current_state = graph.get_state(config)
print("目前暫停在：", current_state.next)
print("目前的計畫：", current_state.values.get("plan"))

approve = input("please check above plan can be approved? (Y/N)")
while approve.upper() == "N":
    try:
        advice = input("please input you advice")
    except Exception as err:
        print(err)
    graph.update_state(config, {"user_advice": advice})
    for event in graph.stream(None, config):
        print(event)

    current_state = graph.get_state(config)
    print("目前暫停在：", current_state.next)
    print("目前的計畫：", current_state.values.get("plan"))
    approve = input("please check above plan can be approved? (Y/N)")


# 3. 人類確認後，傳入 None 恢復執行 ▶️
for event in graph.stream(None, config):
    print(event)

# current_state = graph.get_state(config)
# print("目前暫停在：", current_state.next)
# print("目前的計畫：", current_state.values.get("plan"))