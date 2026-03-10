## 透過
```
=== Phase 1 / Prompt Template ===
好的，這是為期 2 週的 LangChain Model I/O 入門重點規劃：

1.  **第一週：模型基礎與提示工程**
    *   **目標：** 學習 LangChain LLM 與 ChatModel 的初始化及 `invoke()` 使用。掌握 `PromptTemplate` 和 `ChatPromptTemplate` 建立結構化輸入。
    *   **行動：** 實作不同 LLM/ChatModel（如 OpenAI、HuggingFace），並練習設計包含變數的提示模板，理解 System/Human/AI 訊息的角色。

2.  **第二週：輸出處理與基本串聯**
    *   **目標：** 探索 `OutputParser` 處理模型輸出（如 JSON、Pydantic）。學習使用 `|` 運算符串聯 `PromptTemplate`、模型與 `OutputParser`，實現端到端流程。
    *   **行動：** 嘗試 `StrOutputParser`、`JsonOutputParser` 或 `PydanticOutputParser`。建立一個簡單的鏈 (Chain)，將提示、模型和解析器連接起來，並處理其輸出。

3.  **實作與進階 I/O 概念**
    *   **目標：** 透過小型專案整合所學，並理解 `Runnable` 介面。探索 `stream()` 方法以實現即時輸出，提升使用者體驗。
    *   **行動：** 選擇一個簡單的應用場景（如內容摘要、簡單問答），從頭到尾實作一個 LangChain 應用。練習使用 `stream()` 方法觀察模型輸出過程。
```

```
=== Phase 1 / Chat Models vs LLMs ===
[Chat Model]
LangChain Chat Model處理多輪對話，輸入為角色訊息列表；傳統LLM多為單一字串輸入，用於一次性文本生成。

[LLM]
傳統 LLM 呼叫接收單一字串提示，返回單一字串回應，不具備內建上下文記憶。

LangChain 的 Chat Model 則處理結構化訊息列表（如系統、人類、AI），並返回訊息物件。它內建對話歷史管理，更適合構建多輪、有狀態的對話應用，提供更自然的互動體驗。

=== Phase 1 / Output Parsers ===
[JSON Parser]
{
  "title": "掌握 LangChain Prompt Templates",
  "topic": "Prompt Templates",
  "description": "學習如何使用 LangChain 的 Prompt Templates 來創建動態、可重複使用且易於管理的提示，這是構建強大 LLM 應用程式的基礎。",
  "difficulty": "Beginner",
  "estimated_time": "2-4 hours",
  "prerequisites": [
    "Python 基礎",
    "LLM 基本概念"
  ],
  "steps": [
    "理解 Prompt Templates 的核心概念與優勢，包括如何將變數嵌入提示中。",
    "實作 `PromptTemplate` 類別，學習如何定義輸入變數並使用 `.format()` 方法生成最終提示。",
    "探索更進階的模板類型，如 `ChatPromptTemplate`，並了解如何在 LangChain Chains 中整合 Prompt Templates。"
  ],
  "resources": [
    "LangChain 官方文件 (Prompt Templates 章節)",
    "YouTube 教學影片 (搜尋 'LangChain Prompt Templates tutorial')",
    "GitHub 上的 LangChain 範例程式碼"
  ],
  "keywords": [
    "LangChain",
    "Prompt Templates",
    "LLM",
    "AI",
    "Python",
    "Generative AI"
  ]
}

[Pydantic Parser]
{
  "topic": "Output Parsers",
  "difficulty": "intermediate",
  "key_points": [
    "理解 Output Parsers 的核心概念與必要性，為何需要將 LLM 的自由文本輸出轉換為結構化資料。",
    "掌握常見的 Output Parsers 類型（如 PydanticOutputParser, JsonOutputParser, RegexParser）及其使用場景與實作方式。",
    "學習如何處理解析失敗、錯誤處理機制，以及如何客製化 Output Parsers 以滿足特定應用需求。"
  ]
}
```