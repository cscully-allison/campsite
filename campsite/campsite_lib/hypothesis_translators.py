"""Interchangeable hypothesis translation nodes for the analysis state graph.

Each node translates a natural language research question into a formal
hypothesis using varying levels of grammar guidance:

- no_grammar_node: No structural guidance; freeform formalization.
- v1_grammar_node: Uses grammar_v1.txt (expressions + variables).
- v2_grammar_node: Uses grammar_v2.txt (adds estimands).
- v3_grammar_node: Uses grammar_v3.txt (adds explicit uncertainty).

All nodes share the same interface as hypothesis_node and can be swapped
directly into the analysis graph as the hypothesis_manager.
"""

import json
from pathlib import Path
from typing import Callable, Awaitable, Optional

from .utils import AnalysisState, llm, log
from .hypothesis_node import underspecification_check

# Module-level toggle for the underspecification check phase.
# When False (default), nodes skip underspecification checking and
# proceed directly to hypothesis generation.
UNDERSPECIFICATION_CHECK_ENABLED: bool = False

# Directory containing grammar spec files
_GRAMMAR_DIR = Path(__file__).parent / "grammar_specs"


def _load_grammar(filename: str) -> str:
    """Load a grammar specification file from grammar_specs/."""
    return (_GRAMMAR_DIR / filename).read_text()


def _build_question_context(state: AnalysisState) -> str:
    """Build the question context string from state clarifications and refinements."""
    parts: list[str] = [f"Research question: {state.initialUserQuestion}"]

    if state.clarifications:
        parts.append(
            f"Initial clarifications: {'; '.join(state.clarifications)}"
        )

    if state.refinementReport:
        parts.append(
            f"Refinement clarifications: {state.refinementReport} "
            f"(addressing: {'; '.join(state.refinementQuestions)})"
        )

    parts.append(f"Data summary: {json.dumps(state.dataSummary)}")

    return "\n".join(parts)


async def _run_underspecification_phase(
    state: AnalysisState,
) -> Optional[AnalysisState]:
    """Run the underspecification check phase shared by all nodes.

    Skipped entirely when UNDERSPECIFICATION_CHECK_ENABLED is False.

    Returns the modified state if clarification is needed (caller should
    return it immediately), or None if generation should proceed.
    """
    if not UNDERSPECIFICATION_CHECK_ENABLED:
        return None

    if state.substage != "none":
        return None

    clarification_prompts = await underspecification_check(
        state.initialUserQuestion,
        state.clarifications,
        state.dataSummary,
    )
    log(f"Clarifications Required: {json.dumps(clarification_prompts)}")

    if clarification_prompts:
        state.refinementQuestions = clarification_prompts
        state.substage = "refinement"
        state.stage = "conversation"
        return state

    return None


# ---------------------------------------------------------------------------
# Node 1: No-grammar hypothesis translation
# ---------------------------------------------------------------------------


async def no_grammar_node(state: AnalysisState) -> AnalysisState:
    """Hypothesis node with no grammar guidance.

    Asks the LLM to express the research question as a formal hypothesis
    without any structural grammar specification. No contextual information
    is passed forward.
    """
    log("Hypothesis Node (no-grammar) Start\n")

    early_return = await _run_underspecification_phase(state)
    if early_return is not None:
        return early_return

    context = _build_question_context(state)

    response = await llm.ainvoke(
        f"""Express the following research question as a formal statistical hypothesis.
Be precise and unambiguous. Output only the formal hypothesis, nothing else.

{context}
"""
    )

    state.hypothesis = str(response.content)
    state.contextual_information = None
    state.stage = "artifactgen"

    log(
        f"Hypothesis Node (no-grammar) End: {state.hypothesis}\n"
        "-------------------------\n"
    )
    return state


# ---------------------------------------------------------------------------
# Grammar-guided node factory
# ---------------------------------------------------------------------------


def _make_grammar_node(
    grammar_filename: str,
    label: str,
) -> Callable[[AnalysisState], Awaitable[AnalysisState]]:
    """Create a grammar-guided hypothesis translation node.

    The returned node loads the specified grammar, instructs the LLM to
    strictly adhere to it when formalizing the hypothesis, and passes the
    grammar text as contextual_information in the output state.
    """
    grammar_text = _load_grammar(grammar_filename)

    async def grammar_node(state: AnalysisState) -> AnalysisState:
        log(f"Hypothesis Node ({label}) Start\n")

        early_return = await _run_underspecification_phase(state)
        if early_return is not None:
            return early_return

        context = _build_question_context(state)

        response = await llm.ainvoke(
            f"""You are a hypothesis formalization assistant. You must express the
following research question as a formal hypothesis that strictly conforms
to the grammar specification below.

--- GRAMMAR ---
{grammar_text}
--- END GRAMMAR ---

Rules:
1. Your output must be a single hypothesis that is valid according to the
   grammar above. Do not deviate from the grammar's productions.
2. Use the exact syntax defined in the grammar (e.g., comparators, function
   names, predicate notation).
3. Do not include "hypothesis :-" in your output
4. Output ONLY the formal hypothesis expression. No explanation, no
   preamble, no commentary.

{context}
"""
        )

        state.hypothesis = str(response.content)
        state.contextual_information = grammar_text
        state.stage = "artifactgen"

        log(
            f"Hypothesis Node ({label}) End: {state.hypothesis}\n"
            "-------------------------\n"
        )
        return state

    grammar_node.__name__ = f"{label}_grammar_node"
    grammar_node.__qualname__ = f"{label}_grammar_node"
    grammar_node.__doc__ = (
        f"Hypothesis node using {grammar_filename} for grammar-guided translation."
    )

    return grammar_node


# ---------------------------------------------------------------------------
# Nodes 2–4: Grammar-guided hypothesis translation nodes
# ---------------------------------------------------------------------------

v1_grammar_node: Callable[[AnalysisState], Awaitable[AnalysisState]] = (
    _make_grammar_node("grammar_v1_revised.txt", "v1")
)

v2_grammar_node: Callable[[AnalysisState], Awaitable[AnalysisState]] = (
    _make_grammar_node("grammar_v2_revised.txt", "v2")
)

v3_grammar_node: Callable[[AnalysisState], Awaitable[AnalysisState]] = (
    _make_grammar_node("grammar_v3_revised.txt", "v3")
)
