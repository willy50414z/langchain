from __future__ import annotations

import json
import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field


class StudyPlan(BaseModel):
    topic: str = Field(description="學習主題")
    difficulty: str = Field(description="難度等級，僅限 beginner/intermediate/advanced")
    key_points: list[str] = Field(description="3 個重點")


class LangChainService:
    """第一階段(Model I/O) 的 LangChain 範例封裝。"""

    def __init__(
        self,
        chat_model_name: str | None = None,
        llm_model_name: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        load_dotenv()
        self._config = self._load_config(config_path)
        self.base_url = (
            base_url
            or self._read_provider_config("base_url")
            or os.getenv("LOCAL_MODEL_BASE_URL")
            or "http://193.168.1.127:7301/api/generate"
        )
        self.chat_model_name = (
            chat_model_name
            or self._read_provider_config("chat_model")
            or os.getenv("LOCAL_CHAT_MODEL")
            or os.getenv("LOCAL_MODEL_NAME")
            or "gemma4"
        )
        self.llm_model_name = (
            llm_model_name
            or self._read_provider_config("llm_model")
            or os.getenv("LOCAL_LLM_MODEL")
            or self.chat_model_name
        )
        self.temperature = self._resolve_temperature(temperature)
        self.request_timeout = self._resolve_request_timeout()

    @staticmethod
    def _default_config_candidates() -> list[Path]:
        repo_root = Path(__file__).resolve().parents[4]
        return [
            Path.cwd() / "application.ini",
            repo_root / "application.ini",
        ]

    def _load_config(self, config_path: str | Path | None) -> ConfigParser:
        parser = ConfigParser()
        if config_path:
            cfg_path = Path(config_path)
            if cfg_path.exists():
                parser.read(cfg_path, encoding="utf-8")
            return parser

        for candidate in self._default_config_candidates():
            if candidate.exists():
                parser.read(candidate, encoding="utf-8")
                break
        return parser

    def _read_config(self, section: str, key: str) -> str | None:
        if self._config.has_option(section, key):
            value = self._config.get(section, key).strip()
            return value or None
        return None

    def _read_provider_config(self, key: str) -> str | None:
        return self._read_config("local_model", key) or self._read_config("gemini", key)

    def _resolve_temperature(self, temperature: float | None) -> float:
        if temperature is not None:
            return temperature

        value = (
            self._read_provider_config("temperature")
            or os.getenv("LOCAL_MODEL_TEMPERATURE")
            or os.getenv("GEMINI_TEMPERATURE")
        )
        if not value:
            return 0.2

        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Invalid local_model.temperature value in application.ini. Use a number like 0.2.") from exc

    def _resolve_request_timeout(self) -> float:
        value = (
            self._read_provider_config("request_timeout")
            or os.getenv("LOCAL_MODEL_TIMEOUT")
            or "180"
        )
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                "Invalid local_model.request_timeout value in application.ini. Use seconds like 180."
            ) from exc

    def _generate(self, prompt: str, model_name: str | None = None) -> str:
        payload = json.dumps(
            {
                "model": model_name or self.chat_model_name,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Local model API returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise EnvironmentError(
                f"Cannot connect to local model API at {self.base_url}: {exc.reason}"
            ) from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Local model API did not return valid JSON: {body}") from exc

        text = result.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Local model API response missing 'response' text: {result}")
        return text.strip()

    @staticmethod
    def _to_text(result: Any) -> str:
        if hasattr(result, "content"):
            return str(result.content).strip()
        return str(result).strip()

    def demo_prompt_template(self, topic: str, weeks: int = 2) -> str:
        """Prompts: 建立可參數化的 Prompt Template 並呼叫 Chat Model。"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是 LangChain 學習教練，回答要精簡、可執行。"),
                ("human", "請幫我規劃 {weeks} 週的 {topic} 入門重點，使用 3 點條列。"),
            ]
        )
        messages = prompt.format_messages(topic=topic, weeks=weeks)
        rendered_prompt = "\n".join(f"{message.type}: {message.content}" for message in messages)
        return self._generate(rendered_prompt)

    def demo_chat_model_vs_llm(self, question: str) -> dict[str, str]:
        """Chat Models & LLMs: 以 Gemini Chat API 展示 chat 與 llm-style prompt 差異。"""
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是技術助理，回答限制在 80 字內。"),
                ("human", "{question}"),
            ]
        )
        chat_messages = chat_prompt.format_messages(question=question)
        chat_rendered_prompt = "\n".join(f"{message.type}: {message.content}" for message in chat_messages)
        chat_response = self._generate(chat_rendered_prompt)

        llm_prompt = PromptTemplate.from_template(
            "你是技術助理，請用 80 字內回答。\n問題: {question}\n回答:"
        )
        llm_rendered_prompt = llm_prompt.format(question=question)
        llm_response = self._generate(llm_rendered_prompt, model_name=self.llm_model_name)

        return {
            "chat_model": chat_response,
            "llm": llm_response,
        }

    def demo_json_output_parser(self, topic: str) -> dict[str, Any]:
        """Output Parsers: 使用 JsonOutputParser 回傳結構化 JSON。"""
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=(
                "請回傳 JSON，內容是一個 LangChain 學習建議。\n"
                "{format_instructions}\n"
                "主題: {topic}\n"
                "限制: steps 必須有 3 個字串項目。"
            ),
            input_variables=["topic"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        rendered_prompt = prompt.format(topic=topic)
        raw_text = self._generate(rendered_prompt)
        return parser.invoke(raw_text)

    def demo_pydantic_output_parser(self, topic: str) -> StudyPlan:
        """Output Parsers: 使用 PydanticOutputParser 輸出強型別物件。"""
        parser = PydanticOutputParser(pydantic_object=StudyPlan)
        prompt = PromptTemplate(
            template=(
                "根據主題輸出一個學習計畫。\n"
                "{format_instructions}\n"
                "主題: {topic}\n"
                "key_points 需剛好 3 項。"
            ),
            input_variables=["topic"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        rendered_prompt = prompt.format(topic=topic)
        raw_text = self._generate(rendered_prompt)
        return parser.invoke(raw_text)
