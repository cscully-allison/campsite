"""Analysis state graph using LangGraph."""

import json
from langgraph.graph import StateGraph, END, START

from .utils import AnalysisState, llm, log
# from .hypothesis_node import hypothesis_node
from .hypothesis_translators import v3_grammar_node as hypothesis_node
from .artifact_gen import generate_artifact


async def vagueness_check(
    original: str,
    clarifications: list[str] = None,
    dataSummary: dict = None,
) -> int:
    """
    Evaluate the vagueness/meaningfulness of a research question.

    Returns confidence level 0-4 (4 = very high, 0 = very low).
    """
    if clarifications is None:
        clarifications = []

    log("\n---------------------------\n")
    log("Evaluating Vagueness\n")

    clarifications_text = ""
    if clarifications:
        clarifications_text = f"Clarifications: {'; '.join(clarifications)}"

    response = await llm.ainvoke(
        f"""Report your confidence in the following statement:
This research question and the provided clarifications, if present, poses a meaningful analytical question; even if it is underspecified right now.
The research question: {original}

Additional Contextual Information:
{clarifications_text}

Answer with json formatted with the following two fields:
  confidence: An integer between 0-4 representing your confidence level, where 4 is "very high", 3 is "high", 2 is "medium", 1 is "low", and 0 "very low"
  reasoning: a brief explanation of your reasoning."""
    )

    log(f"Vagueness Response: {response.content}\n")

    try:
        result = json.loads(str(response.content))
        return result.get("confidence", 0)
    except json.JSONDecodeError:
        return 0


async def addressed_clarifications_check(
    clarificationQuestions: list[str],
    clarifications: str,
) -> list[str]:
    """
    Check which clarification questions have not been addressed.

    Returns list of unaddressed questions.
    """
    log("Evaluating Addressed Clarifications\n")

    response = await llm.ainvoke(
        f"""I will provide you with a list of questions and natural language response that may answer all, some or none of these questions.

Please compare the natural language response against this list of questions and return any questions which have not been addressed by the response. Ensure that returned questions are phrased exactly how you received them.

Questions: {'; '.join(clarificationQuestions)}
Clarifications: {clarifications}

Return the unaddressed questions as a json formatted list.
"""
    )

    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return clarificationQuestions


async def conversation_node(state: AnalysisState) -> AnalysisState:
    """
    Conversation manager node - handles clarification flow.
    """
    log("Conversation Node Start\n")

    if state.substage == "none":
        log(f"clarification turns: {state.clarificationTurns}\n")

        if state.clarificationTurns > 1:
            log("Max Clarification Turns Reached\n-------------------------\n")
            state.stage = "break"
            state.clarificationNeeded = False
            state.waitingForUser = False
            return state

        confidence = await vagueness_check(
            state.initialUserQuestion,
            state.clarifications,
            state.dataSummary,
        )

        log(f"Vagueness score: {confidence}\n")

        needs_more = confidence < 2  # threshold for meaningful question

        if needs_more:
            log("Clarification Required\n-------------------------\n")

            waiting_text = ""
            if state.waitingForUser and state.userPrompt:
                waiting_text = f"and the question you last asked the user: {state.userPrompt}"

            response = await llm.ainvoke(
                f"""Please respond only with a prompt to the user that asks them to clarify their question, do not mention that you are providing a clarification prompt.
Original Question:
{state.initialUserQuestion}
and keep in mind previous clarifications:
{'; '.join(state.clarifications) if state.clarifications else ''}
and the provided data summary:
{json.dumps(state.dataSummary)}
{waiting_text}
"""
            )

            state.stage = "pause"
            state.waitingForUser = True
            state.userPrompt = str(response.content)
            state.clarificationTurns += 1
            state.turns += 1
            return state

        log("No Clarification Needed\n-------------------------\n")

        state.stage = "hypothesis"
        state.clarificationNeeded = needs_more
        state.waitingForUser = False
        return state

    elif state.substage == "refinement":
        log("Clarification Refinement Start\n")
        log(f"Refinement Convo Context: {json.dumps(state.refinementClarifications)}\n")

        refinement_context = ""
        if state.refinementClarifications:
            refinement_context = "Only tell the user what issues remain to be resolved from the list if they ask or if the conversation context is getting long."
        else:
            refinement_context = "Convey these issues to the user in a nicely formatted list."

        response = await llm.ainvoke(
            f"""You are an assistant that is conveying potential issues with formalizing a user's provided research question.

The following points of ambiguity have been raised by a hypothesis translation component: {'; '.join(state.refinementQuestions)}. {refinement_context}

If a user asks you a question, respond with concise and focused answers to their questions, keeping in mind their original research question and the conversation context.

If the user asks you to provide initial specifications or handle unresolved points of ambiguity, you may suggest specifications for any element which needs disambiguation. Explicitly tell the user what assumptions you will make.

Once you identify that most, or all of the ambiguities have been resolved by your discussion, ask the user to confirm that they would like to develop a formal hypothesis. Only respond with a true for "resolved" if the user has confirmed that they would like you to proceed.

YOU DO NOT FORMALIZE HYPOTHESES YET. YOUR SOLE PURPOSE IS TO RESOLVE AMBIGUITIES.

Current conversation context:
{'; '.join(state.refinementClarifications)}
Their original research question:
{state.initialUserQuestion}
and a summary of the data:
{json.dumps(state.dataSummary)}

Respond with a json object with three elements:
  {{
    "resolved": <a boolean that indicates if ambiguities have been sufficiently resolved>,
    "user_response": <A text based response to the user that addresses the most recent response from the user in the provided conversation context; use markdown formatting where appropriate>,
    "clarifications": <A summary of how the conversation context has clarified points of ambiguity in the hypothesis translation>
  }}
"""
        )

        log(f"\nClarification Refinement conversation: {response.content}\n-------------------------\n")

        try:
            result = json.loads(str(response.content))
        except json.JSONDecodeError:
            result = {"resolved": False, "user_response": "I couldn't process that. Could you try again?", "clarifications": ""}

        if not result.get("resolved", False):
            state.stage = "pause"
            state.waitingForUser = True
            state.userPrompt = result.get("user_response", "")
            state.refinementClarifications = [
                *state.refinementClarifications,
                result.get("user_response", ""),
            ]
            state.clarificationTurns += 1
            state.turns += 1
            return state

        log("Clarification Refinement Done\n-------------------------\n")

        state.stage = "hypothesis"
        state.clarificationNeeded = False
        state.waitingForUser = False
        state.refinementReport = result.get("clarifications", "")
        state.substage = "refined"
        return state

    return state


async def artifact_gen_node(state: AnalysisState) -> AnalysisState:
    """Artifact generation node - LLM generates an Altair visualization."""
    log("Artifact Node Start\n")

    result, code = await generate_artifact(
        hypothesis=state.hypothesis,
        data_summary=state.dataSummary,
        contextual_information=state.contextual_information,
    )

    state.code = code
    state.vega_lite_spec = result.vega_lite_spec
    state.explanation = result.explanation
    state.stage = "done"

    log(f"Artifact Node End\n  code length: {len(code)}\n-------------------------\n")
    return state


def route_after_conversation(state: AnalysisState) -> str:
    """Route after conversation node based on stage."""
    return state.stage


def route_after_hypothesis(state: AnalysisState) -> str:
    """Route after hypothesis node based on stage."""
    return state.stage


def route_after_artifact(state: AnalysisState) -> str:
    """Route after artifact gen node based on stage."""
    return state.stage


# Build the state graph
def create_analysis_graph() -> StateGraph:
    """Create and compile the analysis state graph."""
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("conversation_manager", conversation_node)
    graph.add_node("hypothesis_manager", hypothesis_node)
    graph.add_node("artifact_gen_manager", artifact_gen_node)

    # Add edges
    graph.add_edge(START, "conversation_manager")

    graph.add_conditional_edges(
        "conversation_manager",
        route_after_conversation,
        {
            "hypothesis": "hypothesis_manager",
            "pause": END,
            "break": END,
        },
    )

    graph.add_conditional_edges(
        "hypothesis_manager",
        route_after_hypothesis,
        {
            "conversation": "conversation_manager",  # clarification loop
            "artifactgen": "artifact_gen_manager",
        },
    )

    graph.add_conditional_edges(
        "artifact_gen_manager",
        route_after_artifact,
        {"done": END},
    )

    return graph.compile()


# Compiled analysis assistant graph
analysis_assistant = create_analysis_graph()
