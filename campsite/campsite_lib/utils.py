"""Utility functions and types for Campsite analysis."""

import os
from typing import Literal, Optional, Any, Union
from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

LLMClient = Union[ChatOpenAI, ChatAnthropic]

# Log file path
LOGFILE = "./log.txt"


class AnalysisState(BaseModel):
    """State model for the analysis state machine."""

    stage: Literal[
        "conversation", "hypothesis", "refinement", "artifactgen", "done", "pause", "break"
    ] = "conversation"
    dataSummary: Any = {}

    initialUserQuestion: str = ""
    clarifications: list[str] = []
    refinementQuestions: list[str] = []
    refinementClarifications: list[str] = []
    refinementReport: Optional[str] = None

    hypothesis: Optional[str] = None
    contextual_information: Optional[str] = None
    code: Optional[str] = None
    vega_lite_spec: Optional[str] = None
    explanation: Optional[str] = None

    clarificationNeeded: Optional[bool] = None
    userPrompt: Optional[str] = None

    turns: int = 0
    clarificationTurns: int = 0

    waitingForUser: Optional[bool] = False
    substage: Literal["none", "refinement", "refined"] = "none"

    model_config = ConfigDict(extra="allow")


# Type alias for state type
AnalysisStateType = AnalysisState


def get_llm() -> LLMClient:
    """Get configured LLM instance.

    Provider is selected via the LLM_PROVIDER env var ("anthropic" or
    "openai"), defaulting to "anthropic". API key is read from
    ANTHROPIC_API_KEY or OPENAI_API_KEY accordingly.
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it to use the Campsite analysis features."
            )
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.7,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it to use the Campsite analysis features."
            )
        return ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.7,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Expected 'anthropic' or 'openai'."
    )


# Lazy-loaded LLM instance
_llm_instance: Optional[LLMClient] = None


def get_llm_instance() -> LLMClient:
    """Get or create the singleton LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


# For backwards compatibility - access via property
class LLMProxy:
    """Proxy object for lazy LLM initialization."""

    def __getattr__(self, name):
        return getattr(get_llm_instance(), name)

    def invoke(self, *args, **kwargs):
        return get_llm_instance().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await get_llm_instance().ainvoke(*args, **kwargs)


llm = LLMProxy()


def log(msg: str) -> None:
    """Append message to log file."""
    try:
        with open(LOGFILE, "a") as f:
            f.write(msg)
    except Exception:
        pass  # Silently ignore logging errors


def log_state(state: AnalysisState, note: str) -> None:
    """Log state with a note."""
    log(f"State Log - {note}:\n{state.model_dump_json()}\n-------------------------\n")
