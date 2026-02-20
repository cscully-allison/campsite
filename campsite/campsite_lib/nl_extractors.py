"""Natural Language extractors for semantic invariant checking.

Each extractor is responsible for extracting a single canonical field
from a natural language hypothesis string using an LLM. The NLExtractor
orchestrator runs all field extractors and returns a dict[str, ExtractedValue].
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from .utils import llm, log
from .si_checkers import ExtractedValue


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NLFieldExtractor(ABC):
    """Base class for extracting a single canonical field from NL text."""

    field_id: str = ""
    prompt_template: str = ""

    @abstractmethod
    def normalize(self, raw_value: Any) -> Any:
        """Normalize a raw LLM-extracted value to canonical form."""
        ...

    def extract(self, nl: str) -> ExtractedValue:
        """Extract this field from natural language text using the LLM."""
        if not nl:
            return ExtractedValue(value=None, exists=False)
        if not self.prompt_template:
            return ExtractedValue(value=None, exists=False)

        prompt = self.prompt_template.format(nl=nl)
        try:
            response = llm.invoke(prompt)
            resp = json.loads(response.content)
            raw_value = resp.get("value")
            normalized = self.normalize(raw_value)
            return ExtractedValue(
                value=normalized,
                exists=normalized is not None,
                confidence=resp.get("confidence", 0.8),
                metadata=resp,
            )
        except Exception as e:
            log(f"NL extraction failed for {self.field_id}: {e}\n")
            return ExtractedValue(
                value=None, exists=False,
                metadata={"extraction_failed": True, "error": str(e)},
            )


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------

class ComparatorNLExtractor(NLFieldExtractor):
    """Extract comparator from NL hypothesis."""

    field_id = "event.comparator"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Extract the comparator that describes the relationship between "
        "a quantity of interest and a reference value.\n"
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "<comparator symbol: >, <, =, >=, <=, !=, BETWEEN, or null>", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _NL_ALIASES = {
        "greater than": ">", "more than": ">", "exceeds": ">", "above": ">",
        "higher than": ">",
        "less than": "<", "fewer than": "<", "below": "<", "under": "<",
        "lower than": "<",
        "no more than": "<=", "no less than": ">=",
        "equal to": "=", "equals": "=", "equal": "=",
        "at least": ">=", "greater than or equal to": ">=",
        "at most": "<=", "less than or equal to": "<=",
        "not equal": "!=", "not equal to": "!=", "differs from": "!=",
        "between": "BETWEEN",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class ReferentNLExtractor(NLFieldExtractor):
    """Extract referent from NL hypothesis.

    Classifies the referent as:
    - "threshold" — a fixed numeric value or constant
    - "quantity" — another quantity/measure (triggers sub-extraction of quantity fields)
    - "missing" — no referent stated or implicit
    """

    field_id = "event.referent"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Identify what the main quantity is being compared against (the referent).\n\n"
        "Classify the referent as one of:\n"
        '- "threshold" — a fixed numeric value or constant '
        '(e.g., "5", "100 dollars", "zero")\n'
        '- "quantity" — another quantity or statistical measure '
        '(e.g., "the average salary of managers", "the median score of group B")\n'
        '- "missing" — no referent is stated or it is implicit\n\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "threshold"|"quantity"|"missing", '
        '"referent_quantity_nl": "<if value is \'quantity\', write a concise '
        "natural language description of JUST the referent quantity "
        '(not the main quantity); otherwise null>", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _CATEGORY_ALIASES = {
        "number": "threshold", "numeric": "threshold", "constant": "threshold",
        "fixed": "threshold", "value": "threshold",
        "measure": "quantity", "statistic": "quantity", "metric": "quantity",
        "none": "missing", "implicit": "missing", "unspecified": "missing",
        "null": "missing",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            lowered = raw_value.lower().strip()
            return self._CATEGORY_ALIASES.get(lowered, lowered)
        return raw_value


class SignatureNLExtractor(NLFieldExtractor):
    """Extract quantity signature from NL hypothesis."""

    field_id = "event.quantity.signature"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Classify what kind of quantity is being evaluated:\n"
        '- "level" — a single scalar value describing a data attribute (e.g., "the average salary")\n'
        '- "contrast" — a comparison represented as a difference (e.g., "difference in salary between groups")\n'
        '- "distribution" — a random variable representing uncertainty about a value (e.g., "CI for the mean")\n'
        '- "trend" — slope, change over time (e.g., "increasing trend")\n'
        '- "association" — a symmetric relationship between values (e.g., "correlation between ice cream sales and murders")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "level"|"contrast"|"distribution"|"trend"|"association", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _NL_ALIASES = {
        "mean": "level", "average": "level", "aggregate": "level",
        "difference": "contrast", "comparison": "contrast", "gap": "contrast",
        "correlation": "association", "relationship": "association",
        "slope": "trend", "change": "trend",
        "uncertainty": "distribution", "ci": "distribution",
        "confidence interval": "distribution",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class ConditioningNLExtractor(NLFieldExtractor):
    """Extract conditioning predicates from NL hypothesis."""

    field_id = "event.quantity.conditioning"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Extract all conditioning predicates — subgroup restrictions like "
        '"for engineers", "among women", "when X > 5".\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": [<list of condition descriptions, or empty list>], '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    def normalize(self, raw_value: Any) -> Any:
        return raw_value


class ShapeNLExtractor(NLFieldExtractor):
    """Extract quantity shape from NL hypothesis."""

    field_id = "event.quantity.shape"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Classify the algebraic structure of the quantity:\n"
        '- "value" — a plain aggregate (e.g., "the average of X")\n'
        '- "difference" — a subtraction between groups (e.g., "A minus B")\n'
        '- "ratio" — a ratio between groups (e.g., "ratio of A to B")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "value"|"difference"|"ratio", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _NL_ALIASES = {
        "simple": "value",
        "subtraction": "difference", "division": "ratio",
        "addition": "value", "multiplication": "value",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class UncertaintyNLExtractor(NLFieldExtractor):
    """Extract uncertainty category from NL hypothesis."""

    field_id = "event.quantity.uncertainty"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Determine how uncertainty relates to the quantity:\n"
        '- "missing" — no uncertainty is mentioned\n'
        '- "attached" — uncertainty is directly attached to the quantity '
        '(e.g., "CI for the mean", "bootstrap distribution")\n'
        '- "detached" — uncertainty is mentioned but not structurally attached '
        '(e.g., "there is some uncertainty about...")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "missing"|"attached"|"detached", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _NL_ALIASES = {
        "none": "missing", "no": "missing", "point": "missing",
        "yes": "attached", "distribution": "attached",
        "confidence interval": "attached", "ci": "attached",
        "bootstrap": "attached", "interval": "attached",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class FormNLExtractor(NLFieldExtractor):
    """Extract event form from NL hypothesis."""

    field_id = "event.form"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Classify the form of the event:\n"
        '- "quantity_comp_threshold" — a quantity compared to a fixed threshold\n'
        '- "quantity_comp_quantity" — a quantity compared to another quantity\n'
        '- "quantity_comp_threshold_conditioned" — threshold comparison with conditioning\n'
        '- "quantity_comp_quantity_conditioned" — quantity-vs-quantity with conditioning\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "quantity_comp_threshold"|"quantity_comp_quantity"|'
        '"quantity_comp_threshold_conditioned"|"quantity_comp_quantity_conditioned", '
        '"confidence": <0.0-1.0>, '
        '"rationale": "<short explanation>"}}'
    )

    _NL_ALIASES = {
        "simple": "quantity_comp_threshold",
        "conditional": "quantity_comp_threshold_conditioned",
        "conditioned": "quantity_comp_threshold_conditioned",
        "unconditioned": "quantity_comp_threshold",
        "underspecified": "quantity_comp_threshold",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class NLExtractor:
    """Orchestrates all NL field extractors.

    Runs each field extractor and returns a dict mapping field IDs
    to ExtractedValue results.  When the referent is classified as a
    quantity, quantity-relevant extractors (signature, conditioning,
    shape, uncertainty) are re-run against the referent quantity's NL
    description and stored with ``event.referent.`` prefixed keys.
    """

    def __init__(self):
        self.extractors: list[NLFieldExtractor] = [
            ComparatorNLExtractor(),
            ReferentNLExtractor(),
            SignatureNLExtractor(),
            ConditioningNLExtractor(),
            ShapeNLExtractor(),
            UncertaintyNLExtractor(),
            FormNLExtractor(),
        ]
        # Extractors for quantity-relevant fields (reused for referent-as-quantity)
        self._quantity_extractors: list[NLFieldExtractor] = [
            e for e in self.extractors
            if e.field_id.startswith("event.quantity.")
        ]

    def extract_all(self, nl: str) -> dict[str, ExtractedValue]:
        """Extract all canonical fields from an NL hypothesis.

        Returns a dict mapping field_id to ExtractedValue.
        When the referent is a quantity, additional entries with
        ``event.referent.`` prefixed keys are included for the
        referent quantity's signature, conditioning, shape, and uncertainty.
        """
        results = {}
        for extractor in self.extractors:
            results[extractor.field_id] = extractor.extract(nl)

        # If referent is a quantity, extract quantity fields from referent NL
        referent_result = results.get("event.referent")
        if referent_result and referent_result.value == "quantity":
            referent_nl = referent_result.metadata.get("referent_quantity_nl")
            if referent_nl:
                self._extract_referent_quantity_fields(referent_nl, results)

        return results

    def _extract_referent_quantity_fields(
        self, referent_nl: str, results: dict[str, ExtractedValue]
    ) -> None:
        """Run quantity extractors on the referent quantity's NL description.

        Results are stored in *results* with ``event.referent.`` prefixed keys
        (e.g. ``event.referent.quantity.signature``).
        """
        for extractor in self._quantity_extractors:
            prefixed_key = f"event.referent.{extractor.field_id.removeprefix('event.')}"
            results[prefixed_key] = extractor.extract(referent_nl)

    def get_extractor(self, field_id: str) -> Optional[NLFieldExtractor]:
        """Get a specific extractor by field ID."""
        for extractor in self.extractors:
            if extractor.field_id == field_id:
                return extractor
        return None
