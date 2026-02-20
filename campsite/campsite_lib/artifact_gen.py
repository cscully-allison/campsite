"""Artifact generation: LLM produces an Altair visualization from a hypothesis."""

import json
from typing import Optional

from pydantic import BaseModel, Field

from .utils import get_llm_instance, log


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
    """
    log("Artifact Gen - Generating Altair visualization\n")

    parts = [
        ARTIFACT_GEN_PROMPT,
        f"\nHypothesis: {hypothesis}",
        f"\nDataset Summary:\n{json.dumps(data_summary, indent=2)}",
    ]

    if contextual_information:
        parts.append(
            "\nThe hypothesis above is expressed using a formal grammar. "
            "Here is a description of that grammar to help you interpret it:\n"
            f"{contextual_information}"
        )

    prompt = "\n".join(parts)

    structured_llm = get_llm_instance().with_structured_output(ArtifactResult)
    result = await structured_llm.ainvoke(prompt)

    log(f"Artifact Gen - Spec generated ({len(result.vega_lite_spec)} chars)\n")

    code_response = await get_llm_instance().ainvoke(
        CODE_GEN_PROMPT + result.vega_lite_spec
    )
    code = str(code_response.content)

    log(f"Artifact Gen - Code generated ({len(code)} chars)\n")
    return result, code
