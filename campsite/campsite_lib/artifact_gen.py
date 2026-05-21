"""Artifact generation: LLM produces an Altair visualization from a hypothesis."""

import json
from typing import Optional

from pydantic import BaseModel, Field

from .utils import get_llm_instance, log

ALLOWED_MARKS: set[str] = {"area", "point", "line", "text", "rule", "rect", "bar"}
ALLOWED_CHANNELS: set[str] = {"x", "y", "y2", "color", "row", "column", "text"}
MAX_RETRIES: int = 2


class ArtifactResult(BaseModel):
    """Structured output from the artifact generator."""

    vega_lite_spec: str = Field(
        description="An Altair-friendly vega-lite JSON specification for the visualization, as a JSON string."
    )
    explanation: str = Field(
        description=(
            "A concise explanation of how this visualization helps the user "
            "evaluate the hypothesis."
        )
    )

    @property
    def parsed_vega_lite_spec(self) -> dict:
        """Parse the vega_lite_spec JSON string into a dict."""
        return json.loads(self.vega_lite_spec)


def _extract_marks(spec: dict) -> list[str]:
    """Recursively extract all mark types from a vega-lite spec."""
    marks: list[str] = []
    mark = spec.get("mark")
    if isinstance(mark, str):
        marks.append(mark)
    elif isinstance(mark, dict):
        mark_type = mark.get("type")
        if isinstance(mark_type, str):
            marks.append(mark_type)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        for child in spec.get(key, []):
            if isinstance(child, dict):
                marks.extend(_extract_marks(child))

    return marks


def validate_marks(spec: dict) -> list[str]:
    """Return list of disallowed mark types found in *spec*.

    Returns an empty list when all marks are valid.
    """
    return [m for m in _extract_marks(spec) if m not in ALLOWED_MARKS]


def _extract_encoding_channels(spec: dict) -> list[str]:
    """Recursively extract all encoding channel names from a vega-lite spec."""
    channels: list[str] = []
    encoding = spec.get("encoding")
    if isinstance(encoding, dict):
        channels.extend(encoding.keys())

    for key in ("layer", "hconcat", "vconcat", "concat"):
        for child in spec.get(key, []):
            if isinstance(child, dict):
                channels.extend(_extract_encoding_channels(child))

    return channels


def validate_encodings(spec: dict) -> list[str]:
    """Return list of disallowed encoding channels found in *spec*.

    Returns an empty list when all channels are valid.
    """
    return [ch for ch in _extract_encoding_channels(spec) if ch not in ALLOWED_CHANNELS]


ARTIFACT_GEN_PROMPT = """\
I will pass you a hypothesis specification. Please design a visualization \
that will allow a user to evaluate this hypothesis.

When designing a visualization you will use the encodings: x, y, color, row, and/or column. Prioritize position encodings. \
If a variable you want to visualize has a continuous domain, you **must*** show it \
using either a point or area marking. If the variable is indexed over a time \
series you may use a line.

Effective visualizations may require grouping data for easier comparisions.\

You must provide two things:

1. **vega_lite_spec**: An Altair-friendly vega-lite JSON specification as a JSON string.

2. **explanation**: A concise explanation of how this visualization helps \
evaluate the hypothesis.

Use only column names which are present in the provided dataset summary. \
You may derive metrics from these columns if necessary.
"""

CODE_GEN_PROMPT = """
I will pass a vega-lite JSON specification to you. Please generate a python function containing altair
code that displays a visualization that matches this specification.

Please include this code at the top of the function under the import to altair: alt.data_transformers.enable("vegafusion")

The altair code should not contain any tooltips.
"""


async def generate_artifact(
    hypothesis: str,
    data_summary: dict,
    contextual_information: Optional[str] = None,
) -> tuple[ArtifactResult, str]:
    """Generate an Altair visualization from a hypothesis and data summary.

    Returns a tuple of (ArtifactResult, code) where ArtifactResult contains
    the vega_lite_spec and explanation, and code is the generated Python code.

    The generated spec is validated to ensure only allowed mark types (area,
    point, line) are used.  If validation fails the LLM is re-invoked up to
    ``MAX_RETRIES`` times with explicit feedback about the violation.
    """
    log("Artifact Gen - Generating Altair visualization\n")

    base_parts = [
        ARTIFACT_GEN_PROMPT,
        f"\nHypothesis: {hypothesis}",
        f"\nDataset Summary:\n{json.dumps(data_summary, indent=2)}",
    ]

    if contextual_information:
        base_parts.append(
            "\nThe hypothesis above is expressed using a formal grammar. "
            "Here is a description of that grammar to help you interpret it:\n"
            f"{contextual_information}"
        )

    prompt = "\n".join(base_parts)
    structured_llm = get_llm_instance().with_structured_output(ArtifactResult)

    result: Optional[ArtifactResult] = None
    for attempt in range(1, MAX_RETRIES + 1):
        result = await structured_llm.ainvoke(prompt)
        log(f"Artifact Gen - Spec generated (attempt {attempt}, {len(result.vega_lite_spec)} chars)\n")

        try:
            spec = result.parsed_vega_lite_spec
        except json.JSONDecodeError:
            log("Artifact Gen - Failed to parse vega-lite spec as JSON, retrying\n")
            prompt = "\n".join([
                *base_parts,
                "\n**IMPORTANT**: Your previous response was not valid JSON. "
                "Please return a valid vega-lite JSON specification.",
            ])
            continue

        bad_marks = validate_marks(spec)
        bad_channels = validate_encodings(spec)
        if not bad_marks and not bad_channels:
            break

        feedback_parts = list(base_parts)
        if bad_marks:
            log(f"Artifact Gen - Disallowed marks {bad_marks} found, retrying\n")
            feedback_parts.append(
                f"\n**CRITICAL CONSTRAINT – MARKS**: Your previous visualization used "
                f"the following disallowed mark type(s): {', '.join(bad_marks)}. "
                f"You MUST only use marks from this set: {', '.join(sorted(ALLOWED_MARKS))}. "
                f"Replace any disallowed mark with the closest allowed alternative "
                f"(e.g. use 'point' instead of 'bar' or 'circle')."
            )
        if bad_channels:
            log(f"Artifact Gen - Disallowed channels {bad_channels} found, retrying\n")
            feedback_parts.append(
                f"\n**CRITICAL CONSTRAINT – ENCODINGS**: Your previous visualization used "
                f"the following disallowed encoding channel(s): {', '.join(bad_channels)}. "
                f"You MUST only use encoding channels from this set: "
                f"{', '.join(sorted(ALLOWED_CHANNELS))}. "
                f"Remove any disallowed encoding channels from your specification."
            )
        prompt = "\n".join(feedback_parts)
    else:
        spec = result.parsed_vega_lite_spec
        bad_marks = validate_marks(spec)
        bad_channels = validate_encodings(spec)
        errors: list[str] = []
        if bad_marks:
            errors.append(f"disallowed mark(s): {', '.join(bad_marks)}")
        if bad_channels:
            errors.append(f"disallowed encoding channel(s): {', '.join(bad_channels)}")
        if errors:
            raise ValueError(
                f"Artifact generation failed after {MAX_RETRIES} attempts. "
                f"Spec still contains {'; '.join(errors)}"
            )

    if attempt > 1:
        log(f"Artifact Gen - Spec accepted after {attempt} attempt(s) ({attempt - 1} retry/retries)\n")

    code_response = await get_llm_instance().ainvoke(
        CODE_GEN_PROMPT + result.vega_lite_spec
    )
    code = str(code_response.content)

    log(f"Artifact Gen - Code generated ({len(code)} chars)\n")
    return result, code
