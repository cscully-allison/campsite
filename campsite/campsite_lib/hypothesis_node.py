"""Hypothesis node for the analysis state graph."""

import json
from .utils import AnalysisState, llm, log


async def underspecification_check(
    original_question: str,
    current_clarifications: list[str],
    dataSummary: dict,
) -> list[str]:
    """
    Check if the research question needs further clarification.

    Returns a list of clarification requests, or empty list if none needed.
    """
    log("Underspecification check\n")

    response = await llm.ainvoke(
        f"""This is an initial step before fully generating a formal hypothesis. Using the provided research question and any clarifications provided, identify if any elements may be underspecified.

Do not focus on any implementation details or very low level details at this stage. Only consider points that are critically necessary for an accurate translation from research question to formal hypothesis.

Only return individual underspecifications as a json formatted list of clarification requests.

Original Question: {original_question}
Current Clarifications (may be empty): {'; '.join(current_clarifications)}

Keep in mind that hypotheses may be specified in the context of this dataset, summarized by the following summary.

Relevant Data: {json.dumps(dataSummary)}

It may be the case that no further clarifications are needed. In that case, return a json formatted empty list.
"""
    )

    log(f"Underspecification check done: {response.content}\n")

    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return []


async def hypothesis_node(state: AnalysisState) -> AnalysisState:
    """
    Hypothesis generation node in the state graph.

    Checks for underspecification, requests refinement if needed,
    or generates the formal hypothesis.
    """
    log("Hypothesis Node Start\n")

    if state.substage == "none":
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

    # Generate hypothesis
    clarifications_text = ""
    if state.clarifications:
        clarifications_text = f", and these initial clarifications {'; '.join(state.clarifications)}"

    refinement_text = ""
    if state.refinementReport:
        refinement_text = f"and these refinement clarifications: {state.refinementReport} which answer these identified underspecification ambiguities: {'; '.join(state.refinementQuestions)}"

    response = await llm.ainvoke(
        f"""Generate a formal statistical hypothesis from this question: {state.initialUserQuestion}
            {clarifications_text}
            and this data summary: {json.dumps(state.dataSummary)}
            {refinement_text}
        """
    )

    log(f"Hypothesis Node End {response.content}\n-------------------------\n")
    log("Onto Artifact Gen\n")

    state.hypothesis = str(response.content)
    state.stage = "artifactgen"
    return state
