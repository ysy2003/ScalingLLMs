"""LangChain-powered HTML repair agent utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Part
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_vertexai import ChatVertexAI


@dataclass
class AgentConfig:
    """Configuration block for the LangChain repair agent."""

    project_id: str
    region: str
    model_id: str
    temperature: float = 0.0
    max_output_tokens: int = 9000

    @classmethod
    def from_yaml(cls, payload: Dict) -> "AgentConfig":
        gcp_block = payload.get("gcp", {})
        model_block = payload.get("model", {})
        return cls(
            project_id=gcp_block.get("project_id", ""),
            region=gcp_block.get("region", ""),
            model_id=model_block.get("id", ""),
            temperature=model_block.get("temperature", 0.0),
            max_output_tokens=model_block.get("max_output_tokens", 9000),
        )


class LangChainRepairAgent:
    """Encapsulates LangChain prompt orchestration and Gemini access."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = genai.Client(
            vertexai=True,
            location=config.region,
            project=config.project_id,
            http_options=HttpOptions(api_version="v1"),
        )
        self._llm = ChatVertexAI(
            project=config.project_id,
            location=config.region,
            model_name=config.model_id,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
        self._repair_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert front-end engineer tasked with incrementally "
                        "repairing HTML+CSS artifacts so that they render faithfully while "
                        "improving semantics and accessibility.",
                    ),
                    (
                        "human",
                        "Browser/runtime issues:\n{error_summary}\n\n"
                        "Current quality metrics:\n{metrics_summary}\n\n"
                        "Update the markup while preserving layout intent. Always return the full HTML document.\n"
                        "Broken HTML snippet follows:\n```html\n{broken_html}\n```",
                    ),
                ]
            )
            | self._llm
            | StrOutputParser()
            | RunnableLambda(self._strip_code_fences)
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.lower().startswith("```html"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    @staticmethod
    def _format_errors(errors: Iterable[str]) -> str:
        errors = list(errors)
        return "\n".join(f"- {err}" for err in errors) or "- No runtime errors captured."

    @staticmethod
    def _format_metrics(metrics: Optional[Dict[str, float]]) -> str:
        if not metrics:
            return "- Not available"
        return "\n".join(
            f"- {key}: {value:.3f}" for key, value in metrics.items()
        )

    def repair_html(
        self,
        broken_html: str,
        errors: Iterable[str],
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Invoke the LangChain repair chain and return updated HTML."""
        return self._repair_chain.invoke(
            {
                "error_summary": self._format_errors(errors),
                "metrics_summary": self._format_metrics(metrics),
                "broken_html": broken_html,
            }
        )

    def generate_content(self, contents):
        """Compat helper so legacy functions can request Gemini generations."""
        if isinstance(contents, str):
            contents = [contents]

        response = self._client.models.generate_content(
            model=self.config.model_id,
            contents=contents,
            config=GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        return getattr(response, "text", str(response))

    def build_multimodal_part(self, data: bytes, mime_type: str = "image/png") -> Part:
        """Expose helper for creating Gemini multimedia Part payloads."""
        return Part.from_bytes(data=data, mime_type=mime_type)
