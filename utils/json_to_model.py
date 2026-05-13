"""Helpers to convert text (often LLM output) into pydantic models.

Contains an abstract JSONToModel interface and two implementations:
- LLMJSONToModel: formats the text by asking an LLM (requires an ollama-like client)
- DeterministicJSONToModel: best-effort deterministic extraction and parsing

Also provides a small factory `get_json_to_model`.
"""
import json
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel
from abc import ABC, abstractmethod


class JSONToModel(ABC):
    @abstractmethod
    def to_model(self, json_string: str, type_class: BaseModel) -> BaseModel:
        """Convert the provided string into an instance of `type_class`.

        Args:
            json_string: Raw text (may contain JSON, fenced blocks, or freeform text).
            type_class: A pydantic model class (subclass of BaseModel).

        Returns:
            An instance of `type_class`.
        """
        raise NotImplementedError


class LLMJSONToModel(JSONToModel):
    """Use a chat/LLM client to re-format / validate the JSON then parse.

    The client must provide a `chat` method compatible with the usage below (ollama-like).
    """

    def __init__(self, ollama_client, model_name: str = "gemma3:1b"):
        self.client = ollama_client
        self.model_name = model_name

    def to_model(self, json_string: str, type_class: BaseModel) -> BaseModel:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON formatter. You will receive text that is a JSON object, "
                        "and you will output the same JSON object, ensuring it is valid JSON."
                    ),
                },
                {"role": "user", "content": json_string},
            ],
            format=type_class.model_json_schema(),
        )
        # Expect the model to return a JSON string we can validate directly.
        return type_class.model_validate_json(response.message.content)


class DeterministicJSONToModel(JSONToModel):
    """Parse model output deterministically with heuristics.

    - Try direct JSON parse
    - Try extracting code-fenced JSON blocks
    - Try finding the first balanced-braces block
    - Fallback to simple key:value heuristics
    """

    def to_model(self, json_string: str, type_class: BaseModel) -> BaseModel:
        return self._parse_to_model(json_string, type_class)

    def _extract_json_block(self, text: str) -> Optional[str]:
        text = text.strip()
        # 1) Direct JSON
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # 2) Code-fenced JSON
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            return fence_match.group(1)

        # 3) First balanced braces from the first '{'
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break

        return None

    def _parse_to_model(self, content: str, type_class: BaseModel) -> BaseModel:
        # Try to extract a JSON block
        json_block = self._extract_json_block(content)
        if json_block is not None:
            try:
                data = json.loads(json_block)
                return type_class.model_validate(data)
            except Exception:
                pass

        # Heuristic fallback: parse key: value style lines
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        candidate: Dict[str, Any] = {}
        for line in lines:
            low = line.lower()
            if low.startswith("full_prompt") or low.startswith("prompt"):
                val = line.split(":", 1)[-1].strip().strip('"')
                candidate["full_prompt"] = val
            elif low.startswith("lighting_only_prompt") or low.startswith("lighting_only"):
                val = line.split(":", 1)[-1].strip().strip('"')
                candidate["lighting_only_prompt"] = val

        # If still missing, fallback to using the whole content
        if "full_prompt" not in candidate:
            candidate["full_prompt"] = content.strip()
        if "lighting_only_prompt" not in candidate:
            candidate["lighting_only_prompt"] = candidate["full_prompt"]

        return type_class.model_validate(candidate)


def get_json_to_model(kind: str = "deterministic", ollama_client=None) -> JSONToModel:
    """Factory for JSONToModel implementations.

    kind: 'deterministic' (default) or 'llm'
    """
    k = (kind or "deterministic").lower()
    if k in ("deterministic", "det"):
        return DeterministicJSONToModel()
    if k in ("llm", "ollama"):
        if ollama_client is None:
            raise ValueError("ollama_client must be provided for LLMJSONToModel")
        return LLMJSONToModel(ollama_client)
    raise ValueError(f"Unknown json_to_model kind: {kind}")
