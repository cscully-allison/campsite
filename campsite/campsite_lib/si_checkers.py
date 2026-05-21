"""Semantic Invariant Checkers for hypothesis validation.

Organized around 7 canonical fields that must be preserved across
three representations: NL (Natural Language), IR (Intermediate Representation),
and Artifact (vega-lite visualization specification).

Each field checker extracts its canonical field from IR and Artifact,
receives pre-extracted NL values via a dict, validates existence,
and performs pairwise checks between all representation pairs
(NL→IR, IR→Artifact, NL→Artifact).

Execution is ordered: quantity/comparator/referent checks run first,
then EventFormChecker derives its result from their violations.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .utils import llm, log
from .ir_ast import VALID_COMPARATORS
from . import vega_utils


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ViolationType(str, Enum):
    """Types of semantic violations."""

    MISSING_IN_NL = "missing_in_nl"
    MISSING_IN_IR = "missing_in_ir"
    MISSING_IN_ARTIFACT = "missing_in_artifact"
    NL_IR_MISMATCH = "nl_ir_mismatch"
    IR_ARTIFACT_MISMATCH = "ir_artifact_mismatch"
    NL_ARTIFACT_MISMATCH = "nl_artifact_mismatch"
    MALFORMED = "malformed"


class Criticality(str, Enum):
    """Criticality levels for violations."""

    WARN = "warn"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Semantic invariant sub-field enums
# ---------------------------------------------------------------------------

class ComparatorDirection(str, Enum):
    GREATER = "greater"
    LESS = "less"
    NONE = "none"


class ComparatorStrictness(str, Enum):
    STRICT = "strict"
    INCLUSIVE = "inclusive"
    NA = "na"


class ComparatorPolarity(str, Enum):
    EQUAL = "equal"
    UNEQUAL = "unequal"
    NA = "na"


class ComparatorForm(str, Enum):
    BINARY = "binary"
    RANGE = "range"
    SET = "set"
    UNSPECIFIED = "unspecified"


class ReferentType(str, Enum):
    ABSENT = "absent"
    CONSTANT = "constant"
    COMPUTED = "computed"


class QuantitySignature(str, Enum):
    LEVEL = "level"
    CONTRAST = "contrast"
    ASSOCIATION = "association"
    DISTRIBUTION = "distribution"


class QuantityShape(str, Enum):
    VALUE = "value"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    RANK = "rank"


class QuantityUncertainty(str, Enum):
    MISSING = "missing"
    ATTACHED = "attached"


class ConditionRole(str, Enum):
    FILTER = "filter"
    CONTRAST_ARM = "contrast-arm"
    PARTITION_KEY = "partition-key"
    STRATIFICATION_KEY = "stratification-key"


class ScopeValue(str, Enum):
    NONE = "none"
    GROUPED = "grouped"


class PartitionStructure(str, Enum):
    SINGLE = "single"
    CROSSED = "crossed"
    NESTED = "nested"


class FrameValue(str, Enum):
    NONE = "none"
    STRATIFIED = "stratified"


# Representation mapping constants
NLIR = "NL->IR"
IRARTIFACT = "IR->Artifact"
NLARTIFACT = "NL->Artifact"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """Represents a semantic invariant violation."""

    invariantID: str
    violationType: ViolationType
    message: str
    criticality: Criticality
    expected: Optional[Any] = None
    observed: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "invariantID": self.invariantID,
            "violationType": self.violationType.value,
            "message": self.message,
            "criticality": self.criticality.value,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass
class ExtractedValue:
    """Result of extracting a canonical field from a representation."""

    value: Any = None
    exists: bool = False
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ConditionRecord:
    """A single structured condition record."""

    attr: str = ""
    role: str = ""       # One of ConditionRole values
    values: str = ""     # "each", "{a, b}", "= x", etc.
    ordered: bool = False


# ---------------------------------------------------------------------------
# Comparator decomposition
# ---------------------------------------------------------------------------

_COMPARATOR_DECOMPOSITION: dict[str, dict[str, str]] = {
    ">":       {"direction": "greater", "strictness": "strict",    "polarity": "na",      "form": "binary"},
    "<":       {"direction": "less",    "strictness": "strict",    "polarity": "na",      "form": "binary"},
    ">=":      {"direction": "greater", "strictness": "inclusive", "polarity": "na",      "form": "binary"},
    "<=":      {"direction": "less",    "strictness": "inclusive", "polarity": "na",      "form": "binary"},
    "=":       {"direction": "none",    "strictness": "na",       "polarity": "equal",   "form": "binary"},
    "!=":      {"direction": "none",    "strictness": "na",       "polarity": "unequal", "form": "binary"},
    "BETWEEN": {"direction": "none",    "strictness": "na",       "polarity": "na",      "form": "range"},
    "IN":      {"direction": "none",    "strictness": "na",       "polarity": "na",      "form": "set"},
}


def decompose_comparator(operator: Optional[str]) -> dict[str, str]:
    """Decompose a comparator operator into its 4 sub-fields.

    Returns dict with keys: direction, strictness, polarity, form.
    Returns all-unspecified for unknown/missing operators.
    """
    if operator and operator in _COMPARATOR_DECOMPOSITION:
        return dict(_COMPARATOR_DECOMPOSITION[operator])
    return {"direction": "none", "strictness": "na", "polarity": "na", "form": "unspecified"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIOLATION_TYPE_MAP = {
    "NL": ViolationType.MISSING_IN_NL,
    "IR": ViolationType.MISSING_IN_IR,
    "ART": ViolationType.MISSING_IN_ARTIFACT,
}

_PAIRWISE_VIOLATION_TYPE_MAP = {
    "PW-NLIR": ViolationType.NL_IR_MISMATCH,
    "PW-IRART": ViolationType.IR_ARTIFACT_MISMATCH,
    "PW-NLART": ViolationType.NL_ARTIFACT_MISMATCH,
}


def _dataclass_to_dict(obj) -> dict:
    """Convert dataclass to dictionary recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def _ensure_dict(ir: Any) -> dict:
    """Ensure IR is a plain dict (convert from dataclass if needed)."""
    if ir is None:
        return {}
    if hasattr(ir, "to_dict"):
        return ir.to_dict()
    if hasattr(ir, "__dataclass_fields__"):
        return _dataclass_to_dict(ir)
    return ir


def _collect_predicates(node: dict) -> list[dict]:
    """Walk an event/quantity subtree and collect all predicates as flat dicts."""
    predicates = []
    if not isinstance(node, dict):
        return predicates

    # Direct predicate on this node (conditioned_expr, etc.)
    pred = node.get("predicate")
    if pred and isinstance(pred, dict):
        predicates.extend(_flatten_predicate(pred))

    # Recurse into the quantity
    quantity = node.get("quantity")
    if isinstance(quantity, dict):
        predicates.extend(_collect_predicates(quantity))

    # Recurse into the referent
    referent = node.get("referent")
    if isinstance(referent, dict):
        predicates.extend(_collect_predicates(referent))

    # Recurse into estimand (for rv nodes)
    estimand = node.get("estimand")
    if isinstance(estimand, dict):
        predicates.extend(_collect_predicates(estimand))

    # Recurse into expr (for expectation and conditioned_expr nodes)
    expr_node = node.get("expr")
    if isinstance(expr_node, dict):
        predicates.extend(_collect_predicates(expr_node))

    # Recurse into lhs/rhs (for contrast and binary nodes)
    for side in ("lhs", "rhs"):
        child = node.get(side)
        if isinstance(child, dict):
            predicates.extend(_collect_predicates(child))

    return predicates


def _flatten_predicate(pred: dict) -> list[dict]:
    """Flatten a possibly-conjunctive predicate into a list of simple predicates."""
    if pred.get("kind") in ("conjunction", "disjunction"):
        parts = []
        if pred.get("lhs"):
            parts.extend(_flatten_predicate(pred["lhs"]))
        if pred.get("rhs"):
            parts.extend(_flatten_predicate(pred["rhs"]))
        return parts
    return [{"attr": pred.get("attr"), "comparator": pred.get("comparator"), "value": pred.get("value")}]


def _has_conditioning(quantity: dict) -> bool:
    """Check if a quantity subtree has any predicates."""
    return len(_collect_predicates(quantity)) > 0


def _is_error_node(node: Any) -> bool:
    """Check if a node is an ErrorNode (dict with type='error')."""
    return isinstance(node, dict) and node.get("type") == "error"


def _malformed_extracted(node: dict) -> ExtractedValue:
    """Create an ExtractedValue marking a malformed subtree."""
    return ExtractedValue(
        value=None, exists=False,
        metadata={
            "malformed": True,
            "boundary": node.get("boundary", ""),
            "message": node.get("message", ""),
        },
    )


def _extract_partition_attrs(partition: dict) -> list[str]:
    """Extract attribute names from a partition or cross-partition dict."""
    if not isinstance(partition, dict):
        return []
    if partition.get("type") == "partition":
        attr = partition.get("attr", "")
        return [attr] if attr else []
    if partition.get("type") == "cross_partition":
        lhs = _extract_partition_attrs(partition.get("lhs", {}))
        rhs = _extract_partition_attrs(partition.get("rhs", {}))
        return lhs + rhs
    return []


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CanonicalFieldChecker(ABC):
    """Base class for canonical field semantic invariant checkers.

    Each subclass checks one canonical field across NL, IR, and Artifact
    representations. NL values are provided externally as pre-extracted
    ExtractedValue objects via the nl_values dict.
    """

    field_id: str = ""
    required_in: tuple[str, ...] = ()  # Which representations MUST contain this field
    confidence_threshold: float = 0.5  # Below this, pairwise violations downgrade to WARN

    # -- Extractors --

    @abstractmethod
    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Extract this canonical field from the IR dict. Must be implemented."""
        ...

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Extract this canonical field from a vega-lite artifact spec.

        Stub — returns not-found. Override when artifact extraction is implemented.
        """
        return ExtractedValue(value=None, exists=False)

    # -- Checks --

    def check_existence(self, representation: str, extracted: ExtractedValue) -> Optional[Violation]:
        """Check whether the canonical field exists in the given representation."""
        if representation not in self.required_in:
            return None
        if extracted.metadata.get("malformed"):
            boundary = extracted.metadata.get("boundary", "")
            msg = extracted.metadata.get("message", "")
            return Violation(
                invariantID=f"{self.field_id}-MALFORMED-{representation}",
                violationType=ViolationType.MALFORMED,
                message=f"Canonical field '{self.field_id}' could not be checked: "
                        f"malformed {boundary} in {representation}"
                        + (f" ({msg})" if msg else ""),
                criticality=Criticality.FAIL,
                expected="well-formed subtree",
                observed=f"ErrorNode(boundary={boundary})",
            )
        if not extracted.exists:
            return Violation(
                invariantID=f"{self.field_id}-EX-{representation}",
                violationType=_VIOLATION_TYPE_MAP.get(representation, ViolationType.MALFORMED),
                message=f"Canonical field '{self.field_id}' not found in {representation}",
                criticality=Criticality.WARN,
                expected="present",
                observed=extracted.value,
            )
        return None

    def check_pairwise(
        self,
        source: ExtractedValue,
        target: ExtractedValue,
        pair_label: str,
    ) -> Optional[Violation]:
        """Compare the canonical field between two representations.

        Default implementation does equality comparison.
        Subclasses can override for fuzzy / semantic matching.
        """
        if not source.exists or not target.exists:
            return None  # Can't compare if one side is missing

        if not self._values_match(source.value, target.value):
            low_confidence = (
                source.confidence < self.confidence_threshold
                or target.confidence < self.confidence_threshold
            )
            return Violation(
                invariantID=f"{self.field_id}-{pair_label}",
                violationType=_PAIRWISE_VIOLATION_TYPE_MAP.get(
                    pair_label, ViolationType.NL_IR_MISMATCH
                ),
                message=f"Canonical field '{self.field_id}' differs between representations",
                criticality=Criticality.WARN if low_confidence else Criticality.FAIL,
                expected=source.value,
                observed=target.value,
            )
        return None

    def _values_match(self, source_value: Any, target_value: Any) -> bool:
        """Compare two extracted values. Override for custom matching logic."""
        return source_value == target_value

    # -- Orchestrator --

    def check(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, "ExtractedValue"]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run all checks for this canonical field.

        1. Extract from each available representation
        2. Run existence checks
        3. Run pairwise checks between all representation pairs
        """
        violations = []
        ir_dict = _ensure_dict(ir)

        # Get NL value from pre-extracted dict
        nl_val = (nl_values or {}).get(self.field_id, ExtractedValue())

        # Extract from IR and artifact
        ir_val = self.extract_from_ir(ir_dict) if ir_dict else ExtractedValue()
        art_val = self.extract_from_artifact(artifact) if artifact else ExtractedValue()

        # Existence checks
        has_nl = nl_val.exists or (nl_values is not None and self.field_id in nl_values)
        if has_nl:
            v = self.check_existence("NL", nl_val)
            if v:
                violations.append(v)

        if ir_dict:
            v = self.check_existence("IR", ir_val)
            if v:
                violations.append(v)

        if artifact:
            v = self.check_existence("ART", art_val)
            if v:
                violations.append(v)

        # Pairwise checks: NL→IR, IR→ART, NL→ART
        if has_nl and ir_dict:
            v = self.check_pairwise(nl_val, ir_val, "PW-NLIR")
            if v:
                violations.append(v)

        if ir_dict and artifact:
            v = self.check_pairwise(ir_val, art_val, "PW-IRART")
            if v:
                violations.append(v)

        if has_nl and artifact:
            v = self.check_pairwise(nl_val, art_val, "PW-NLART")
            if v:
                violations.append(v)

        return violations


# ---------------------------------------------------------------------------
# Concrete field checkers
# ---------------------------------------------------------------------------

# Sub-field keys for decomposed checkers
_COMPARATOR_SUBFIELDS = ("direction", "strictness", "polarity", "form")
_REFERENT_SUBFIELDS = ("type", "value")
_PARTITION_SUBFIELDS = ("structure", "ordered")


class ComparatorChecker(CanonicalFieldChecker):
    """comparator — Decomposed directional relation.

    Sub-invariants: direction, strictness, polarity, form.
    Extracted as operator from IR, then decomposed programmatically.
    Produces individual violations per sub-field mismatch.
    """

    field_id = "comparator"
    required_in = ("IR",)

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        comparator = event.get("comparator")
        decomposed = decompose_comparator(comparator)
        exists = comparator is not None and (
            comparator in VALID_COMPARATORS or comparator == "\u16A6"
        )
        return ExtractedValue(
            value=decomposed, exists=exists,
            metadata={"raw_operator": comparator},
        )

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Vega-lite specs do not encode directional comparators."""
        return ExtractedValue(
            value=decompose_comparator(None), exists=True,
        )

    def check_pairwise(
        self, source: ExtractedValue, target: ExtractedValue, pair_label: str,
    ) -> Optional[list[Violation]]:
        """Compare each sub-field independently. Skip artifact pairs."""
        if "ART" in pair_label:
            return None
        if not source.exists or not target.exists:
            return None
        if not isinstance(source.value, dict) or not isinstance(target.value, dict):
            return super().check_pairwise(source, target, pair_label)

        violations = []
        for key in _COMPARATOR_SUBFIELDS:
            s_val = source.value.get(key)
            t_val = target.value.get(key)
            if s_val != t_val:
                low_confidence = (
                    source.confidence < self.confidence_threshold
                    or target.confidence < self.confidence_threshold
                )
                violations.append(Violation(
                    invariantID=f"comparator.{key}-{pair_label}",
                    violationType=_PAIRWISE_VIOLATION_TYPE_MAP.get(
                        pair_label, ViolationType.NL_IR_MISMATCH
                    ),
                    message=f"Comparator sub-field '{key}' differs between representations",
                    criticality=Criticality.WARN if low_confidence else Criticality.FAIL,
                    expected=s_val,
                    observed=t_val,
                ))
        return violations if violations else None

    def check(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, "ExtractedValue"]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run checks, flattening per-sub-field pairwise results."""
        violations = []
        ir_dict = _ensure_dict(ir)

        nl_val = (nl_values or {}).get(self.field_id, ExtractedValue())
        ir_val = self.extract_from_ir(ir_dict) if ir_dict else ExtractedValue()
        art_val = self.extract_from_artifact(artifact) if artifact else ExtractedValue()

        # Existence checks
        has_nl = nl_val.exists or (nl_values is not None and self.field_id in nl_values)
        if has_nl:
            v = self.check_existence("NL", nl_val)
            if v:
                violations.append(v)
        if ir_dict:
            v = self.check_existence("IR", ir_val)
            if v:
                violations.append(v)
        if artifact:
            v = self.check_existence("ART", art_val)
            if v:
                violations.append(v)

        # Pairwise checks (returns list or None)
        for src, tgt, label in [
            (nl_val, ir_val, "PW-NLIR"),
            (ir_val, art_val, "PW-IRART"),
            (nl_val, art_val, "PW-NLART"),
        ]:
            if label == "PW-NLIR" and not (has_nl and ir_dict):
                continue
            if label == "PW-IRART" and not (ir_dict and artifact):
                continue
            if label == "PW-NLART" and not (has_nl and artifact):
                continue
            result = self.check_pairwise(src, tgt, label)
            if isinstance(result, list):
                violations.extend(result)
            elif result is not None:
                violations.append(result)

        return violations


class ReferentChecker(CanonicalFieldChecker):
    """referent — Reference value defining the event boundary.

    Sub-invariants: type (absent|constant|computed), value (literal or ᚦ).
    Produces individual violations per sub-field mismatch.
    """

    field_id = "referent"
    required_in = ("IR",)

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        referent = event.get("referent")
        if referent is None:
            return ExtractedValue(value={"type": "absent", "value": "\u16A6"}, exists=True)
        if _is_error_node(referent):
            return _malformed_extracted(referent)
        if isinstance(referent, dict):
            if referent.get("type") == "unspecified":
                return ExtractedValue(
                    value={"type": "absent", "value": "\u16A6"},
                    exists=True, metadata={"unspecified": True},
                )
            if referent.get("type") == "const":
                return ExtractedValue(
                    value={"type": "constant", "value": referent.get("value")},
                    exists=True,
                )
            if referent.get("type") == "func":
                return ExtractedValue(
                    value={"type": "computed", "value": referent.get("name", "\u16A6")},
                    exists=True,
                )
            # Any other type — treat as computed for forward compatibility
            return ExtractedValue(
                value={"type": "computed", "value": "\u16A6"},
                exists=True, metadata={"referent_data": referent},
            )
        return ExtractedValue(value={"type": "constant", "value": referent}, exists=True)

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Detect referent type from vega-lite spec structural patterns."""
        pattern = vega_utils.detect_referent_pattern(artifact)
        type_map = {"threshold": "constant", "quantity": "computed", "missing": "absent"}
        ref_type = type_map.get(pattern, "absent")
        return ExtractedValue(
            value={"type": ref_type, "value": "\u16A6"},
            exists=True,
            metadata={"detection_method": "structural_pattern"},
        )

    def check_pairwise(
        self, source: ExtractedValue, target: ExtractedValue, pair_label: str,
    ) -> Optional[list[Violation]]:
        """Compare type and value sub-fields independently."""
        if not source.exists or not target.exists:
            return None
        if not isinstance(source.value, dict) or not isinstance(target.value, dict):
            return super().check_pairwise(source, target, pair_label)

        violations = []
        for key in _REFERENT_SUBFIELDS:
            s_val = source.value.get(key)
            t_val = target.value.get(key)
            if s_val != t_val:
                low_confidence = (
                    source.confidence < self.confidence_threshold
                    or target.confidence < self.confidence_threshold
                )
                violations.append(Violation(
                    invariantID=f"referent.{key}-{pair_label}",
                    violationType=_PAIRWISE_VIOLATION_TYPE_MAP.get(
                        pair_label, ViolationType.NL_IR_MISMATCH
                    ),
                    message=f"Referent sub-field '{key}' differs between representations",
                    criticality=Criticality.WARN if low_confidence else Criticality.FAIL,
                    expected=s_val,
                    observed=t_val,
                ))
        return violations if violations else None

    def check(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, "ExtractedValue"]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run checks, flattening per-sub-field pairwise results."""
        violations = []
        ir_dict = _ensure_dict(ir)

        nl_val = (nl_values or {}).get(self.field_id, ExtractedValue())
        ir_val = self.extract_from_ir(ir_dict) if ir_dict else ExtractedValue()
        art_val = self.extract_from_artifact(artifact) if artifact else ExtractedValue()

        has_nl = nl_val.exists or (nl_values is not None and self.field_id in nl_values)
        if has_nl:
            v = self.check_existence("NL", nl_val)
            if v:
                violations.append(v)
        if ir_dict:
            v = self.check_existence("IR", ir_val)
            if v:
                violations.append(v)
        if artifact:
            v = self.check_existence("ART", art_val)
            if v:
                violations.append(v)

        for src, tgt, label in [
            (nl_val, ir_val, "PW-NLIR"),
            (ir_val, art_val, "PW-IRART"),
            (nl_val, art_val, "PW-NLART"),
        ]:
            if label == "PW-NLIR" and not (has_nl and ir_dict):
                continue
            if label == "PW-IRART" and not (ir_dict and artifact):
                continue
            if label == "PW-NLART" and not (has_nl and artifact):
                continue
            result = self.check_pairwise(src, tgt, label)
            if isinstance(result, list):
                violations.extend(result)
            elif result is not None:
                violations.append(result)

        return violations


class QuantitySignatureChecker(CanonicalFieldChecker):
    """quantity.signature — What kind of quantity is being evaluated.

    Values: level | contrast | association | distribution.
    Trend has been removed; temporal trends are now a composition of
    level + grouped scope + ordered partition + directional comparator.
    """

    field_id = "quantity.signature"
    required_in = ("IR",)

    _SIGNATURE_MAP = {
        "expectation": "level",
        "contrast": "contrast",
        "rv": "distribution",
        "extract": "level",
    }

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        quantity = event.get("quantity", {}) or {}
        if _is_error_node(quantity):
            return _malformed_extracted(quantity)
        qtype = quantity.get("type")

        # Special case: func node with CORR → association
        if qtype == "func" and quantity.get("name") == "CORR":
            return ExtractedValue(value="association", exists=True)

        sig = self._SIGNATURE_MAP.get(qtype, "unknown")
        return ExtractedValue(value=sig, exists=(sig != "unknown"))

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Classify quantity signature from vega-lite spec.

        Priority: area+quant→distribution,
        point scatter→association, point strip→distribution,
        line/area+temporal→level (was trend, now composition),
        grouping→contrast, default→level.
        """
        units = vega_utils.walk_units(artifact)
        if not units:
            return ExtractedValue(value=None, exists=False)

        marks = vega_utils.extract_all_marks(artifact)

        # Line mark → level (trend is now a composition, not a signature)
        if "line" in marks:
            return ExtractedValue(value="level", exists=True)

        # Area marks
        if "area" in marks:
            for unit in units:
                if vega_utils.get_mark_type(unit) == "area":
                    enc = vega_utils.get_encodings(unit)
                    x_type = vega_utils.get_encoding_type(enc.get("x", {}))
                    if x_type == "temporal":
                        return ExtractedValue(value="level", exists=True)
                    if x_type == "quantitative":
                        return ExtractedValue(value="distribution", exists=True)

        # Point marks
        if "point" in marks:
            for unit in units:
                if vega_utils.get_mark_type(unit) == "point":
                    enc = vega_utils.get_encodings(unit)
                    x_def = enc.get("x", {})
                    y_def = enc.get("y", {})
                    x_type = vega_utils.get_encoding_type(x_def)
                    y_type = vega_utils.get_encoding_type(y_def)
                    x_agg = vega_utils.get_encoding_aggregate(x_def)
                    y_agg = vega_utils.get_encoding_aggregate(y_def)

                    # Both axes quantitative, no aggregation → association
                    if (x_type == "quantitative" and y_type == "quantitative"
                            and not x_agg and not y_agg):
                        return ExtractedValue(value="association", exists=True)

                    # Nominal/ordinal x, quantitative y, no y aggregation → distribution
                    if (x_type in ("nominal", "ordinal")
                            and y_type == "quantitative" and not y_agg):
                        return ExtractedValue(value="distribution", exists=True)

        # Grouping or multiple primary units → contrast
        primary_units = [
            u for u in units
            if vega_utils.get_mark_type(u) not in ("rule", "text", None)
        ]
        if vega_utils.has_grouping(artifact) or len(primary_units) > 1:
            return ExtractedValue(value="contrast", exists=True)

        # Default: single aggregated view → level
        return ExtractedValue(value="level", exists=True)


class ConditionsChecker(CanonicalFieldChecker):
    """conditions — Structured condition records with roles.

    Each condition is {attr, role, values, ordered}.
    IR signals:
    - Predicates on ConditionedExpr → filter or contrast-arm
    - Canonical binding rule: identical predicate on all exprs in contrast = filter
    - ACROSS partition → partition-key
    - WITHIN partition → stratification-key
    Produces per-condition, per-sub-field violations.
    """

    field_id = "conditions"
    required_in = ()  # Conditions are optional

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Extract conditions from IR with role assignment."""
        conditions: list[dict] = []
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)

        quantity = event.get("quantity", {}) or {}
        predicates = _collect_predicates(event)

        # Determine roles for predicates
        is_contrast = isinstance(quantity, dict) and quantity.get("type") == "contrast"
        if is_contrast:
            lhs_preds = _collect_predicates(quantity.get("lhs", {}))
            rhs_preds = _collect_predicates(quantity.get("rhs", {}))
            lhs_attrs = {p.get("attr") for p in lhs_preds}
            rhs_attrs = {p.get("attr") for p in rhs_preds}
            shared_attrs = lhs_attrs & rhs_attrs

            for pred in predicates:
                attr = pred.get("attr", "")
                role = "filter" if attr in shared_attrs else "contrast-arm"
                conditions.append({
                    "attr": attr,
                    "role": role,
                    "values": str(pred.get("value", "")),
                    "ordered": False,
                })
        else:
            for pred in predicates:
                conditions.append({
                    "attr": pred.get("attr", ""),
                    "role": "filter",
                    "values": str(pred.get("value", "")),
                    "ordered": False,
                })

        # Extract partition-key from across_partition
        across = ir.get("across_partition")
        if isinstance(across, dict):
            for part_attr in _extract_partition_attrs(across):
                conditions.append({
                    "attr": part_attr,
                    "role": "partition-key",
                    "values": "each",
                    "ordered": False,
                })

        # Extract stratification-key from within_partition
        within = ir.get("within_partition")
        if isinstance(within, dict):
            for part_attr in _extract_partition_attrs(within):
                conditions.append({
                    "attr": part_attr,
                    "role": "stratification-key",
                    "values": "each",
                    "ordered": False,
                })

        return ExtractedValue(value=conditions, exists=len(conditions) > 0)

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Extract conditions from vega-lite spec with role assignment."""
        conditions: list[dict] = []
        seen_fields: set[str] = set()

        # Filter transforms → filter role
        for ft in vega_utils.get_filter_transforms(artifact):
            filter_val = ft.get("filter")
            if isinstance(filter_val, dict):
                field_name = filter_val.get("field", "?")
                if field_name not in seen_fields:
                    conditions.append({
                        "attr": field_name,
                        "role": "filter",
                        "values": str(filter_val.get("equal", filter_val.get("gte", ""))),
                        "ordered": False,
                    })
                    seen_fields.add(field_name)
            elif isinstance(filter_val, str) and filter_val not in seen_fields:
                conditions.append({
                    "attr": filter_val,
                    "role": "filter",
                    "values": "",
                    "ordered": False,
                })
                seen_fields.add(filter_val)

        # Color grouping fields → partition-key
        for field_name in vega_utils.get_color_grouping_fields(artifact):
            if field_name not in seen_fields:
                conditions.append({
                    "attr": field_name,
                    "role": "partition-key",
                    "values": "each",
                    "ordered": False,
                })
                seen_fields.add(field_name)

        # Facet fields → stratification-key
        for field_name in vega_utils.get_facet_fields(artifact):
            if field_name not in seen_fields:
                conditions.append({
                    "attr": field_name,
                    "role": "stratification-key",
                    "values": "each",
                    "ordered": False,
                })
                seen_fields.add(field_name)

        return ExtractedValue(
            value=conditions,
            exists=len(conditions) > 0,
            metadata={"source": "artifact_structural"},
        )

    def check_pairwise(
        self, source: ExtractedValue, target: ExtractedValue, pair_label: str,
    ) -> Optional[list[Violation]]:
        """Match conditions by attr, then compare sub-fields independently."""
        if not source.exists or not target.exists:
            return None
        if not isinstance(source.value, list) or not isinstance(target.value, list):
            return super().check_pairwise(source, target, pair_label)

        violations = []
        low_confidence = (
            source.confidence < self.confidence_threshold
            or target.confidence < self.confidence_threshold
        )
        crit = Criticality.WARN if low_confidence else Criticality.FAIL
        vtype = _PAIRWISE_VIOLATION_TYPE_MAP.get(pair_label, ViolationType.NL_IR_MISMATCH)

        # Index conditions by attr
        s_by_attr = {c.get("attr", ""): c for c in source.value if isinstance(c, dict)}
        t_by_attr = {c.get("attr", ""): c for c in target.value if isinstance(c, dict)}

        all_attrs = set(s_by_attr.keys()) | set(t_by_attr.keys())
        for attr in sorted(all_attrs):
            s_cond = s_by_attr.get(attr)
            t_cond = t_by_attr.get(attr)

            if s_cond and not t_cond:
                # Present in source but missing in target
                target_rep = pair_label.split("-")[-1]  # e.g., "NLIR" → last part
                violations.append(Violation(
                    invariantID=f"conditions[{attr}]-MISSING-{target_rep}",
                    violationType=vtype,
                    message=f"Condition on '{attr}' present in source but missing in target",
                    criticality=crit,
                    expected=s_cond,
                    observed=None,
                ))
                continue
            if t_cond and not s_cond:
                source_rep = pair_label.split("-")[-1][:2] if len(pair_label.split("-")) > 1 else ""
                violations.append(Violation(
                    invariantID=f"conditions[{attr}]-MISSING-{source_rep}",
                    violationType=vtype,
                    message=f"Condition on '{attr}' present in target but missing in source",
                    criticality=crit,
                    expected=None,
                    observed=t_cond,
                ))
                continue

            # Both present — compare sub-fields
            for key in ("role", "values", "ordered"):
                s_val = s_cond.get(key)
                t_val = t_cond.get(key)
                if s_val != t_val:
                    violations.append(Violation(
                        invariantID=f"conditions[{attr}].{key}-{pair_label}",
                        violationType=vtype,
                        message=f"Condition '{attr}' sub-field '{key}' differs",
                        criticality=crit,
                        expected=s_val,
                        observed=t_val,
                    ))

        return violations if violations else None

    def check(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, "ExtractedValue"]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run checks, flattening per-condition pairwise results."""
        violations = []
        ir_dict = _ensure_dict(ir)

        nl_val = (nl_values or {}).get(self.field_id, ExtractedValue())
        ir_val = self.extract_from_ir(ir_dict) if ir_dict else ExtractedValue()
        art_val = self.extract_from_artifact(artifact) if artifact else ExtractedValue()

        has_nl = nl_val.exists or (nl_values is not None and self.field_id in nl_values)

        for src, tgt, label in [
            (nl_val, ir_val, "PW-NLIR"),
            (ir_val, art_val, "PW-IRART"),
            (nl_val, art_val, "PW-NLART"),
        ]:
            if label == "PW-NLIR" and not (has_nl and ir_dict):
                continue
            if label == "PW-IRART" and not (ir_dict and artifact):
                continue
            if label == "PW-NLART" and not (has_nl and artifact):
                continue
            result = self.check_pairwise(src, tgt, label)
            if isinstance(result, list):
                violations.extend(result)
            elif result is not None:
                violations.append(result)

        return violations


class ShapeChecker(CanonicalFieldChecker):
    """quantity.shape — Algebraic structure of the quantity.

    Categories: value, difference, ratio, rank.
    """

    field_id = "quantity.shape"
    required_in = ("IR",)

    _OP_SHAPE_MAP = {
        "-": "difference",
        "\u2212": "difference",  # −
        "/": "ratio",
        "+": "value",
        "*": "value",
    }

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        quantity = event.get("quantity", {}) or {}
        if _is_error_node(quantity):
            return _malformed_extracted(quantity)
        qtype = quantity.get("type")

        if qtype == "expectation":
            return ExtractedValue(value="value", exists=True)

        if qtype == "contrast":
            op = quantity.get("op", "-")
            shape = self._OP_SHAPE_MAP.get(op, "difference")
            return ExtractedValue(value=shape, exists=True)

        if qtype == "rv":
            estimand = quantity.get("estimand", {}) or {}
            if _is_error_node(estimand):
                return _malformed_extracted(estimand)
            if estimand.get("type") == "contrast":
                op = estimand.get("op", "-")
                # Flatten nested shapes: nested_difference → difference, nested_ratio → ratio
                return ExtractedValue(value=self._OP_SHAPE_MAP.get(op, "difference"), exists=True)
            return ExtractedValue(value="value", exists=True)

        if qtype == "func":
            return ExtractedValue(value="value", exists=True)

        if qtype == "extract":
            return ExtractedValue(value="value", exists=True)

        return ExtractedValue(value="unknown", exists=False)

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Detect quantity shape from vega-lite spec.

        Calculate transform with division → ratio.
        Composition (concat/multi-layer) → difference.
        Default → value.
        """
        if vega_utils.has_division_calculate(artifact):
            return ExtractedValue(value="ratio", exists=True)

        has_concat = bool(
            artifact.get("hconcat") or artifact.get("vconcat")
            or artifact.get("concat")
        )
        if has_concat:
            return ExtractedValue(value="difference", exists=True)

        units = vega_utils.walk_units(artifact)
        if len(units) > 1:
            primary_marks = [
                vega_utils.get_mark_type(u) for u in units
                if vega_utils.get_mark_type(u) not in ("rule", "text", None)
            ]
            if len(primary_marks) >= 2:
                return ExtractedValue(value="difference", exists=True)

        return ExtractedValue(value="value", exists=True)


class UncertaintyChecker(CanonicalFieldChecker):
    """quantity.uncertainty — How uncertainty relates to the quantity.

    Categories: missing (no uncertainty), attached (RV wrapping quantity).
    """

    field_id = "quantity.uncertainty"
    required_in = ("IR",)

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        quantity = event.get("quantity", {}) or {}
        if _is_error_node(quantity):
            return _malformed_extracted(quantity)
        is_rv = quantity.get("type") == "rv"
        return ExtractedValue(
            value="attached" if is_rv else "missing", exists=True
        )

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Detect uncertainty encoding in vega-lite spec.

        Area/rule marks with y+y2 → attached. Otherwise → missing.
        """
        if vega_utils.has_uncertainty_encoding(artifact):
            return ExtractedValue(value="attached", exists=True)
        return ExtractedValue(value="missing", exists=True)


class ScopeChecker(CanonicalFieldChecker):
    """scope — Whether the quantity is distributed across partition levels.

    Values: none | grouped.
    """

    field_id = "scope"
    required_in = ()

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Scope is 'grouped' when across_partition is present."""
        across = ir.get("across_partition")
        scope = "grouped" if across is not None else "none"
        return ExtractedValue(value=scope, exists=True)

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Detect scope from vega-lite spec."""
        if vega_utils.has_grouping(artifact):
            return ExtractedValue(value="grouped", exists=True)
        if len(vega_utils.get_facet_fields(artifact)) > 0:
            return ExtractedValue(value="grouped", exists=True)
        return ExtractedValue(value="none", exists=True)


class PartitionChecker(CanonicalFieldChecker):
    """partition — Structure and ordering of the partition (when scope=grouped).

    Sub-invariants:
    - partition.structure: single | crossed | nested
    - partition.ordered: true | false
    Produces individual violations per sub-field mismatch.
    """

    field_id = "partition"
    required_in = ()

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Extract partition structure from IR."""
        across = ir.get("across_partition")
        if across is None:
            return ExtractedValue(value=None, exists=False)
        if isinstance(across, dict):
            if across.get("type") == "cross_partition":
                structure = "crossed"
            else:
                structure = "single"
        else:
            structure = "single"
        return ExtractedValue(
            value={"structure": structure, "ordered": False},
            exists=True,
        )

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Extract partition structure from vega-lite spec.

        Uses color grouping fields and aggregate groupby fields to
        determine how many grouping dimensions exist.
        """
        dimensions: set[str] = set()
        for f in vega_utils.get_color_grouping_fields(artifact):
            dimensions.add(f)
        for f in vega_utils.get_aggregate_groupby_fields(artifact):
            dimensions.add(f)

        if not dimensions:
            return ExtractedValue(value=None, exists=False)

        structure = "single" if len(dimensions) == 1 else "crossed"
        ordered = vega_utils.detect_partition_ordering(artifact)
        return ExtractedValue(
            value={"structure": structure, "ordered": ordered},
            exists=True,
        )

    def check_pairwise(
        self, source: ExtractedValue, target: ExtractedValue, pair_label: str,
    ) -> Optional[list[Violation]]:
        """Compare partition sub-fields independently."""
        if not source.exists or not target.exists:
            return None
        if not isinstance(source.value, dict) or not isinstance(target.value, dict):
            return super().check_pairwise(source, target, pair_label)

        violations = []
        for key in _PARTITION_SUBFIELDS:
            s_val = source.value.get(key)
            t_val = target.value.get(key)
            if s_val != t_val:
                low_confidence = (
                    source.confidence < self.confidence_threshold
                    or target.confidence < self.confidence_threshold
                )
                violations.append(Violation(
                    invariantID=f"partition.{key}-{pair_label}",
                    violationType=_PAIRWISE_VIOLATION_TYPE_MAP.get(
                        pair_label, ViolationType.NL_IR_MISMATCH
                    ),
                    message=f"Partition sub-field '{key}' differs between representations",
                    criticality=Criticality.WARN if low_confidence else Criticality.FAIL,
                    expected=s_val,
                    observed=t_val,
                ))
        return violations if violations else None

    def check(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, "ExtractedValue"]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run checks, flattening per-sub-field pairwise results."""
        violations = []
        ir_dict = _ensure_dict(ir)

        nl_val = (nl_values or {}).get(self.field_id, ExtractedValue())
        ir_val = self.extract_from_ir(ir_dict) if ir_dict else ExtractedValue()
        art_val = self.extract_from_artifact(artifact) if artifact else ExtractedValue()

        has_nl = nl_val.exists or (nl_values is not None and self.field_id in nl_values)

        for src, tgt, label in [
            (nl_val, ir_val, "PW-NLIR"),
            (ir_val, art_val, "PW-IRART"),
            (nl_val, art_val, "PW-NLART"),
        ]:
            if label == "PW-NLIR" and not (has_nl and ir_dict):
                continue
            if label == "PW-IRART" and not (ir_dict and artifact):
                continue
            if label == "PW-NLART" and not (has_nl and artifact):
                continue
            result = self.check_pairwise(src, tgt, label)
            if isinstance(result, list):
                violations.extend(result)
            elif result is not None:
                violations.append(result)

        return violations


class FrameChecker(CanonicalFieldChecker):
    """frame — Whether stratification is applied.

    Values: none | stratified.
    Observable when within_partition is present.
    """

    field_id = "frame"
    required_in = ()

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Frame is 'stratified' when within_partition is present."""
        within = ir.get("within_partition")
        frame = "stratified" if within is not None else "none"
        return ExtractedValue(value=frame, exists=True)

    def extract_from_artifact(self, artifact: dict) -> ExtractedValue:
        """Detect frame from vega-lite spec.

        Faceted layouts indicate stratification.
        """
        facets = vega_utils.get_facet_fields(artifact)
        if len(facets) > 0:
            return ExtractedValue(value="stratified", exists=True)
        return ExtractedValue(value="none", exists=True)


# ---------------------------------------------------------------------------
# Runner / Orchestrator
# ---------------------------------------------------------------------------

class SICheckRunner:
    """Orchestrates all semantic invariant checks.

    No phased execution — all checkers are independent and run in a flat loop.
    """

    def __init__(self):
        self._all_checkers: list[CanonicalFieldChecker] = [
            ComparatorChecker(),
            ReferentChecker(),
            QuantitySignatureChecker(),
            ShapeChecker(),
            UncertaintyChecker(),
            ConditionsChecker(),
            ScopeChecker(),
            PartitionChecker(),
            FrameChecker(),
        ]

    def run_all(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, ExtractedValue]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run all semantic invariant checks."""
        ir_dict = _ensure_dict(ir)
        violations: list[Violation] = []
        for checker in self._all_checkers:
            violations.extend(
                checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact)
            )
        return violations

    def run_ir_only(self, ir: Any) -> list[Violation]:
        """Run checks using only the IR representation."""
        return self.run_all(ir=ir)

    def run_field(
        self,
        field_id: str,
        ir: Any = None,
        nl_values: Optional[dict[str, ExtractedValue]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run checks for a specific canonical field."""
        for checker in self._all_checkers:
            if checker.field_id == field_id:
                return checker.check(ir=ir, nl_values=nl_values, artifact=artifact)
        return []

    def get_checker(self, field_id: str) -> Optional[CanonicalFieldChecker]:
        """Get a specific checker by field ID."""
        for checker in self._all_checkers:
            if checker.field_id == field_id:
                return checker
        return None
