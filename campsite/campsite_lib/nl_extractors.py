"""Natural Language extractors for semantic invariant checking.

Each extractor is responsible for extracting a single canonical field
from a natural language hypothesis string using an LLM. The NLExtractor
orchestrator runs all field extractors and returns a dict[str, ExtractedValue].
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from .utils import llm, log
from .si_checkers import ExtractedValue


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _llm_content_to_str(content: Any) -> str:
    """Coerce a langchain response.content to a string.

    Anthropic responses may arrive as a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]) when extra modalities
    are in play; join the text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


def parse_llm_json(content: Any) -> dict:
    """Parse JSON from an LLM response, tolerating common wrappers.

    Handles bare JSON, JSON inside ```json``` fences, and JSON preceded
    or followed by prose. Raises json.JSONDecodeError if nothing parses.
    """
    text = _llm_content_to_str(content).strip()
    if not text:
        raise json.JSONDecodeError("empty LLM response", text or "", 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("no parsable JSON in LLM response", text, 0)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NLFieldExtractor(ABC):
    """Base class for extracting a single canonical field from NL text."""

    field_id: str = ""
    prompt_template: str = ""

    def __init__(self, include_rationale: bool = True):
        self.include_rationale = include_rationale

    def _format_prompt(self, nl: str) -> str:
        rationale = ', "rationale": "<short explanation>"' if self.include_rationale else ""
        return self.prompt_template.format(nl=nl, rationale=rationale)

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

        prompt = self._format_prompt(nl)
        response = None
        try:
            response = llm.invoke(prompt)
            resp = parse_llm_json(response.content)
            raw_value = resp.get("value")
            normalized = self.normalize(raw_value)
            kwargs: dict[str, Any] = dict(
                value=normalized,
                exists=normalized is not None,
                confidence=resp.get("confidence", 0.8),
            )
            if self.include_rationale:
                kwargs["metadata"] = resp
            return ExtractedValue(**kwargs)
        except Exception as e:
            raw = _llm_content_to_str(response.content) if response is not None else ""
            log(f"NL extraction failed for {self.field_id}: {e}\nraw: {raw!r}\n")
            return ExtractedValue(
                value=None, exists=False,
                metadata={"extraction_failed": True, "error": str(e), "raw": raw},
            )


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------

class ComparatorNLExtractor(NLFieldExtractor):
    """Extract comparator from NL hypothesis as a raw operator symbol."""

    field_id = "comparator"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Extract the comparator that describes the relationship between "
        "a quantity of interest and a reference value.\n"
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "<comparator symbol: >, <, =, >=, <=, !=, BETWEEN, or null>", '
        '"confidence": <0.0-1.0>{rationale}}}'
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
        return None


class ReferentNLExtractor(NLFieldExtractor):
    """Extract referent from NL hypothesis.

    Returns a dict with:
    - "type": absent | constant | computed
    - "value": the literal threshold value or ᚦ when unclear
    """

    field_id = "referent"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Identify what the main quantity is being compared against (the referent).\n\n"
        "Classify the referent type as one of:\n"
        '- "absent" — no referent is stated or it is implicit\n'
        '- "constant" — a fixed literal value '
        '(e.g., "5", "100 dollars", "zero")\n'
        '- "computed" — a derived/computed value like a statistical summary '
        '(e.g., "the median", "the population average", "the baseline rate")\n\n'
        "Also extract the referent value: the literal number, string, or description "
        "(e.g., 50000, \"median\", \"national average\"). "
        "Use the thorn character '\u16A6' if the value is unclear or absent.\n\n"
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"type": "absent"|"constant"|"computed", '
        '"value": <the referent value or "\u16A6">, '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _TYPE_ALIASES = {
        "threshold": "constant", "number": "constant", "numeric": "constant",
        "fixed": "constant", "literal": "constant",
        "derived": "computed", "calculated": "computed", "summary": "computed",
        "statistic": "computed", "measure": "computed", "quantity": "computed",
        "none": "absent", "implicit": "absent", "unspecified": "absent",
        "missing": "absent", "null": "absent",
    }

    def normalize(self, raw_value: Any) -> Any:
        # normalize is not used directly; extract() is overridden
        return raw_value

    def extract(self, nl: str) -> ExtractedValue:
        """Override to extract both type and value from LLM response."""
        if not nl:
            return ExtractedValue(value=None, exists=False)

        prompt = self._format_prompt(nl)
        response = None
        try:
            response = llm.invoke(prompt)
            resp = parse_llm_json(response.content)
            raw_type = resp.get("type", "absent")
            if isinstance(raw_type, str):
                ref_type = self._TYPE_ALIASES.get(raw_type.lower().strip(), raw_type.lower().strip())
            else:
                ref_type = "absent"
            ref_value = resp.get("value", "\u16A6")
            kwargs: dict[str, Any] = dict(
                value={"type": ref_type, "value": ref_value},
                exists=ref_type is not None,
                confidence=resp.get("confidence", 0.8),
            )
            if self.include_rationale:
                kwargs["metadata"] = resp
            return ExtractedValue(**kwargs)
        except Exception as e:
            raw = _llm_content_to_str(response.content) if response is not None else ""
            log(f"NL extraction failed for {self.field_id}: {e}\nraw: {raw!r}\n")
            return ExtractedValue(
                value=None, exists=False,
                metadata={"extraction_failed": True, "error": str(e), "raw": raw},
            )


class SignatureNLExtractor(NLFieldExtractor):
    """Extract quantity signature from NL hypothesis."""

    field_id = "quantity.signature"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Classify what kind of quantity is being evaluated:\n"
        '- "level" — a single scalar value describing a data attribute '
        '(e.g., "the average salary", "an increasing trend over time")\n'
        '- "contrast" — a comparison represented as a difference '
        '(e.g., "difference in salary between groups")\n'
        '- "distribution" — a random variable representing uncertainty '
        'about a value (e.g., "CI for the mean")\n'
        '- "association" — a symmetric relationship between values '
        '(e.g., "correlation between ice cream sales and murders")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "level"|"contrast"|"distribution"|"association", '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _NL_ALIASES = {
        "mean": "level", "average": "level", "aggregate": "level",
        "trend": "level", "slope": "level", "change": "level",
        "difference": "contrast", "comparison": "contrast", "gap": "contrast",
        "correlation": "association", "relationship": "association",
        "uncertainty": "distribution", "ci": "distribution",
        "confidence interval": "distribution",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class ConditionsNLExtractor(NLFieldExtractor):
    """Extract structured condition records from NL hypothesis.

    Each condition is a record with:
    - attr: the data attribute
    - role: filter | contrast-arm | partition-key | stratification-key
    - values: value specification ("each", "{a, b}", "= x")
    - ordered: whether levels have intrinsic order
    """

    field_id = "conditions"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Extract all conditions — data attributes that restrict, partition, "
        "or stratify the analysis.\n\n"
        "For each condition, determine:\n"
        '1. "attr": the data attribute name (e.g., "department", "age_group")\n'
        '2. "role": one of:\n'
        '   - "filter": restricts to a subset '
        '(signals: "among X", "where X = y", "for X only", "restricted to")\n'
        '   - "contrast-arm": defines groups being compared '
        '(signals: "X vs Y", "differs from", "between A and B")\n'
        '   - "partition-key": splits data into groups for omnibus analysis '
        '(signals: "by X", "across X", "for each X", "grouped by")\n'
        '   - "stratification-key": repeats analysis independently within strata '
        '(signals: "within each X", "separately for each X", "stratified by")\n'
        '3. "values": value specification — '
        '"each" for all levels, specific values like "{{a, b}}", '
        'or "= x" for filters\n'
        '4. "ordered": true if levels have intrinsic order '
        "(time, dose, rank), false otherwise\n\n"
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": [{{' '"attr": "...", "role": "...", "values": "...", "ordered": true|false'
        "}}], "
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, list):
            return [
                {
                    "attr": c.get("attr", ""),
                    "role": c.get("role", "filter"),
                    "values": str(c.get("values", "")),
                    "ordered": bool(c.get("ordered", False)),
                }
                for c in raw_value
                if isinstance(c, dict)
            ]
        return raw_value


class ShapeNLExtractor(NLFieldExtractor):
    """Extract quantity shape from NL hypothesis."""

    field_id = "quantity.shape"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Classify the algebraic structure of the quantity:\n"
        '- "value" — a plain aggregate (e.g., "the average of X")\n'
        '- "difference" — a subtraction between groups (e.g., "A minus B")\n'
        '- "ratio" — a ratio between groups (e.g., "ratio of A to B")\n'
        '- "rank" — a rank ordering (e.g., "ranked by X", "percentile position")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "value"|"difference"|"ratio"|"rank", '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _NL_ALIASES = {
        "simple": "value",
        "subtraction": "difference", "division": "ratio",
        "addition": "value", "multiplication": "value",
        "ranking": "rank", "percentile": "rank", "ordinal": "rank",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class UncertaintyNLExtractor(NLFieldExtractor):
    """Extract uncertainty category from NL hypothesis."""

    field_id = "quantity.uncertainty"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Determine how uncertainty relates to the quantity:\n"
        '- "missing" — no uncertainty is mentioned\n'
        '- "attached" — uncertainty is directly attached to the quantity '
        '(e.g., "CI for the mean", "bootstrap distribution", '
        '"margin of error")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "missing"|"attached", '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _NL_ALIASES = {
        "none": "missing", "no": "missing", "point": "missing",
        "detached": "missing",
        "yes": "attached", "distribution": "attached",
        "confidence interval": "attached", "ci": "attached",
        "bootstrap": "attached", "interval": "attached",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class ScopeNLExtractor(NLFieldExtractor):
    """Extract scope from NL hypothesis.

    Values: none | grouped.
    """

    field_id = "scope"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Determine whether the quantity is evaluated across groups "
        "or as a single aggregate:\n"
        '- "none" — the hypothesis describes a single overall quantity\n'
        '- "grouped" — the quantity is distributed across partition levels '
        '(signals: "across groups", "by category", "for each level", '
        '"grouped by")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "none"|"grouped", '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _NL_ALIASES = {
        "single": "none", "overall": "none", "aggregate": "none",
        "partitioned": "grouped", "split": "grouped", "stratified": "grouped",
    }

    def normalize(self, raw_value: Any) -> Any:
        if isinstance(raw_value, str):
            return self._NL_ALIASES.get(raw_value.lower().strip(), raw_value)
        return raw_value


class PartitionNLExtractor(NLFieldExtractor):
    """Extract partition structure from NL hypothesis.

    Returns dict with:
    - "structure": single | crossed | nested
    - "ordered": true | false
    """

    field_id = "partition"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "If the hypothesis involves grouping or partitioning, classify:\n\n"
        "1. Structure:\n"
        '- "single" — one grouping dimension (e.g., "by department")\n'
        '- "crossed" — Cartesian product of two dimensions '
        '(e.g., "by gender and age group")\n'
        '- "nested" — hierarchical grouping '
        '(e.g., "by classroom within school")\n'
        "If no partitioning is present, return null.\n\n"
        "2. Ordered: true if the partition levels have intrinsic order "
        "(temporal, dose, rank), false otherwise.\n\n"
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"structure": "single"|"crossed"|"nested"|null, '
        '"ordered": true|false, '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    def normalize(self, raw_value: Any) -> Any:
        return raw_value

    def extract(self, nl: str) -> ExtractedValue:
        """Override to extract both structure and ordered from LLM response."""
        if not nl:
            return ExtractedValue(value=None, exists=False)

        prompt = self._format_prompt(nl)
        response = None
        try:
            response = llm.invoke(prompt)
            resp = parse_llm_json(response.content)
            structure = resp.get("structure")
            if structure is None:
                if self.include_rationale:
                    return ExtractedValue(value=None, exists=False, metadata=resp)
                return ExtractedValue(value=None, exists=False)
            ordered = bool(resp.get("ordered", False))
            kwargs: dict[str, Any] = dict(
                value={"structure": structure, "ordered": ordered},
                exists=True,
                confidence=resp.get("confidence", 0.8),
            )
            if self.include_rationale:
                kwargs["metadata"] = resp
            return ExtractedValue(**kwargs)
        except Exception as e:
            raw = _llm_content_to_str(response.content) if response is not None else ""
            log(f"NL extraction failed for {self.field_id}: {e}\nraw: {raw!r}\n")
            return ExtractedValue(
                value=None, exists=False,
                metadata={"extraction_failed": True, "error": str(e), "raw": raw},
            )


class FrameNLExtractor(NLFieldExtractor):
    """Extract frame from NL hypothesis.

    Values: none | stratified.
    """

    field_id = "frame"
    prompt_template = (
        "You will be provided with a natural language hypothesis.\n"
        "Determine whether the hypothesis is repeated independently "
        "within strata:\n"
        '- "none" — no stratification\n'
        '- "stratified" — the analysis is repeated within strata '
        '(signals: "within each X", "separately for each X", '
        '"stratified by X")\n'
        "Natural Language Hypothesis: {nl}\n\n"
        "Return a JSON object:\n"
        '{{"value": "none"|"stratified", '
        '"confidence": <0.0-1.0>{rationale}}}'
    )

    _NL_ALIASES = {
        "unstratified": "none", "overall": "none",
        "repeated": "stratified", "faceted": "stratified",
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
    to ExtractedValue results.
    """

    def __init__(self, include_rationale: bool = True):
        self.include_rationale = include_rationale
        self.extractors: list[NLFieldExtractor] = [
            ComparatorNLExtractor(include_rationale=include_rationale),
            ReferentNLExtractor(include_rationale=include_rationale),
            SignatureNLExtractor(include_rationale=include_rationale),
            ShapeNLExtractor(include_rationale=include_rationale),
            UncertaintyNLExtractor(include_rationale=include_rationale),
            ConditionsNLExtractor(include_rationale=include_rationale),
            # ScopeNLExtractor(include_rationale=include_rationale),
            # PartitionNLExtractor(include_rationale=include_rationale),
            # FrameNLExtractor(include_rationale=include_rationale),
        ]

    def extract_all(self, nl: str) -> dict[str, ExtractedValue]:
        """Extract all canonical fields from an NL hypothesis.

        Returns a dict mapping field_id to ExtractedValue.
        """
        results = {}
        for extractor in self.extractors:
            results[extractor.field_id] = extractor.extract(nl)
        return results

    def get_extractor(self, field_id: str) -> Optional[NLFieldExtractor]:
        """Get a specific extractor by field ID."""
        for extractor in self.extractors:
            if extractor.field_id == field_id:
                return extractor
        return None
