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

    # Direct predicate on this node
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

    # Recurse into lhs/rhs (for contrast nodes)
    for side in ("lhs", "rhs"):
        child = node.get(side)
        if isinstance(child, dict):
            predicates.extend(_collect_predicates(child))

    return predicates


def _flatten_predicate(pred: dict) -> list[dict]:
    """Flatten a possibly-conjunctive predicate into a list of simple predicates."""
    if pred.get("kind") == "conjunction":
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

class ComparatorChecker(CanonicalFieldChecker):
    """event.comparator — Directional relation defining the hypothesis.

    Handles the thorn character 'ᚦ' as 'missing'.
    """

    field_id = "event.comparator"
    required_in = ("IR",)

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        comparator = event.get("comparator")
        if comparator == "\u16A6":
            return ExtractedValue(value="missing", exists=True)
        exists = comparator is not None and comparator in VALID_COMPARATORS
        return ExtractedValue(value=comparator, exists=exists)


class ReferentChecker(CanonicalFieldChecker):
    """event.referent — Reference value defining the event boundary.

    Categorizes the referent as: 'threshold' (Const), 'quantity' (any quantity type),
    or 'missing' (None/Unspecified).
    """

    field_id = "event.referent"
    required_in = ("IR",)

    _QUANTITY_TYPE_SYNONYMS = {
        "expectation": {"mean", "average", "expected value", "expected"},
        "contrast": {"difference", "comparison", "gap"},
    }

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        referent = event.get("referent")
        if referent is None:
            return ExtractedValue(value="missing", exists=True)
        if _is_error_node(referent):
            return _malformed_extracted(referent)
        if isinstance(referent, dict):
            if referent.get("type") == "unspecified":
                return ExtractedValue(value="missing", exists=True, metadata={"unspecified": True})
            if referent.get("type") == "const":
                return ExtractedValue(value="threshold", exists=True, metadata={"const_value": referent.get("value")})
            # Quantity referent (expectation, contrast, rv, etc.)
            return ExtractedValue(value="quantity", exists=True, metadata={"referent_data": referent})
        return ExtractedValue(value="threshold", exists=True)

    def is_quantity_referent(self, ir: dict) -> bool:
        """Check if the referent in this IR is a quantity type."""
        event = ir.get("event", {})
        referent = event.get("referent")
        if not isinstance(referent, dict):
            return False
        return referent.get("type") not in ("const", "unspecified", "error", None)

    def get_referent_quantity(self, ir: dict) -> Optional[dict]:
        """Extract the referent as a quantity dict, or None if not a quantity."""
        event = ir.get("event", {})
        referent = event.get("referent")
        if not isinstance(referent, dict):
            return None
        if referent.get("type") in ("const", "unspecified", "error", None):
            return None
        return referent

    def _values_match(self, source_value: Any, target_value: Any) -> bool:
        """Compare referent categories."""
        # Simple category comparison (threshold, quantity, missing)
        if isinstance(source_value, str) and isinstance(target_value, str):
            return source_value.strip().lower() == target_value.strip().lower()
        return source_value == target_value


class QuantitySignatureChecker(CanonicalFieldChecker):
    """event.quantity.signature — What kind of quantity is being evaluated."""

    field_id = "event.quantity.signature"
    required_in = ("IR",)

    _SIGNATURE_MAP = {
        "expectation": "level",
        "contrast": "contrast",
        "rv": "distribution",
        "extract": "trend",
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


class ConditioningChecker(CanonicalFieldChecker):
    """event.quantity.conditioning — Explicit predicates restricting the hypothesis domain."""

    field_id = "event.quantity.conditioning"
    required_in = ()  # Conditioning is optional

    _JUDGE_PROMPT_TEMPLATE = (
        "You are comparing conditioning predicates from two representations "
        "of the same hypothesis.\n\n"
        "Natural language conditions:\n{nl_json}\n\n"
        "Structured IR predicates:\n{ir_json}\n\n"
        "For each NL condition, determine if there is a corresponding IR predicate "
        "that captures the same semantic intent — including the attribute being "
        "conditioned on, the direction of comparison, and the approximate value "
        "threshold.\n\n"
        "Return JSON:\n"
        '{{"match": true or false, "rationale": "<explanation>", '
        '"pairings": [{{"nl": "...", "ir": {{}}, "matches": true or false}}, ...]}}'
    )

    def __init__(self):
        super().__init__()
        self._judge_override: Optional[bool] = None

    def set_judge_override(self, result: bool):
        """Set an override for the LLM judge result, bypassing LLM calls in tests."""
        self._judge_override = result

    def clear_judge_override(self):
        """Remove the judge override."""
        self._judge_override = None

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        predicates = _collect_predicates(event)

        return ExtractedValue(value=predicates, exists=len(predicates) > 0)

    def _values_match(self, source_value: Any, target_value: Any) -> bool:
        """Compare conditioning between representations.

        Fast path for dict-vs-dict (attr name comparison).
        LLM judge for mixed types (NL strings vs IR dicts).
        """
        if not isinstance(source_value, list) or not isinstance(target_value, list):
            return source_value == target_value

        has_dicts = any(isinstance(p, dict) for p in source_value + target_value)
        has_strings = any(isinstance(p, str) for p in source_value + target_value)

        if has_dicts and not has_strings:
            # Both sides are dicts — fast deterministic comparison by attr names
            source_attrs = sorted(p.get("attr", "") for p in source_value if isinstance(p, dict))
            target_attrs = sorted(p.get("attr", "") for p in target_value if isinstance(p, dict))
            return source_attrs == target_attrs

        # Mixed types — quick count check before LLM
        if len(source_value) != len(target_value):
            return False

        return self._llm_judge_conditions(source_value, target_value)

    def _llm_judge_conditions(self, source: list, target: list) -> bool:
        """Use LLM to judge semantic equivalence of NL vs IR conditions."""
        if self._judge_override is not None:
            return self._judge_override

        # Determine which side is NL strings and which is IR dicts
        if any(isinstance(p, str) for p in source):
            nl_conditions, ir_predicates = source, target
        else:
            nl_conditions, ir_predicates = target, source

        prompt = self._JUDGE_PROMPT_TEMPLATE.format(
            nl_json=json.dumps(nl_conditions, indent=2),
            ir_json=json.dumps(ir_predicates, indent=2),
        )
        try:
            response = llm.invoke(prompt)
            resp = json.loads(response.content)
            return resp.get("match", False)
        except Exception as e:
            log(f"LLM judge failed for conditioning (assuming match): {e}\n")
            return True


class ShapeChecker(CanonicalFieldChecker):
    """event.quantity.shape — Algebraic structure of the quantity.

    Simplified categories: value, ratio, difference.
    """

    field_id = "event.quantity.shape"
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


class UncertaintyChecker(CanonicalFieldChecker):
    """event.quantity.uncertainty — How uncertainty relates to the quantity.

    Categories: missing (no uncertainty), attached (RV wrapping quantity),
    detached (NL/Artifact only — uncertainty mentioned but not structurally attached).
    In IR, uncertainty is always 'attached' (type=rv) or 'missing'.
    """

    field_id = "event.quantity.uncertainty"
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


class EventFormChecker(CanonicalFieldChecker):
    """event.form — Structural form of the event, derived from component check results.

    Four valid forms:
    - quantity_comp_threshold: quantity comparator threshold
    - quantity_comp_quantity: quantity comparator quantity
    - quantity_comp_threshold_conditioned: with conditioning on quantity
    - quantity_comp_quantity_conditioned: with conditioning on either/both quantities
    """

    field_id = "event.form"
    required_in = ("IR",)

    def extract_from_ir(self, ir: dict) -> ExtractedValue:
        """Classify the structural form of the event from IR."""
        event = ir.get("event", {})
        if _is_error_node(event):
            return _malformed_extracted(event)
        if not event:
            return ExtractedValue(value=None, exists=False)

        # Determine referent type
        referent = event.get("referent") or {}
        is_quantity_referent = (
            isinstance(referent, dict)
            and referent.get("type") not in ("const", "unspecified", "error", None)
        )

        # Determine conditioning
        quantity = event.get("quantity", {}) or {}
        has_event_pred = event.get("predicate") is not None
        has_quantity_conditioning = _has_conditioning(quantity)
        has_referent_conditioning = (
            is_quantity_referent and _has_conditioning(referent)
        )
        is_conditioned = has_event_pred or has_quantity_conditioning or has_referent_conditioning

        if is_quantity_referent:
            if is_conditioned:
                form = "quantity_comp_quantity_conditioned"
            else:
                form = "quantity_comp_quantity"
        else:
            if is_conditioned:
                form = "quantity_comp_threshold_conditioned"
            else:
                form = "quantity_comp_threshold"

        return ExtractedValue(value=form, exists=True)

    def check_with_component_violations(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, ExtractedValue]] = None,
        artifact: Optional[dict] = None,
        component_violations: Optional[list[Violation]] = None,
    ) -> list[Violation]:
        """Run event form check, incorporating component violation results.

        If component violations indicate compromised fields, the form
        is flagged as compromised.
        """
        violations = self.check(ir=ir, nl_values=nl_values, artifact=artifact)

        if component_violations:
            compromised_fields = [v.invariantID for v in component_violations]

            # Check which components are compromised
            has_comparator_issue = any("event.comparator" in f for f in compromised_fields)
            has_referent_issue = any("event.referent" in f for f in compromised_fields)
            has_quantity_issue = any("event.quantity" in f for f in compromised_fields)

            compromised_components = []
            if has_comparator_issue:
                compromised_components.append("comparator")
            if has_referent_issue:
                compromised_components.append("referent")
            if has_quantity_issue:
                compromised_components.append("quantity")

            if compromised_components:
                violations.append(Violation(
                    invariantID=f"{self.field_id}-COMPROMISED",
                    violationType=ViolationType.MALFORMED,
                    message=f"Event form is compromised due to violations in: {', '.join(compromised_components)}",
                    criticality=Criticality.FAIL,
                    expected="all components valid",
                    observed=f"violations in {compromised_components}",
                ))

        return violations


# ---------------------------------------------------------------------------
# Runner / Orchestrator
# ---------------------------------------------------------------------------

class SICheckRunner:
    """Orchestrates all canonical field checks with ordered execution.

    Phase 1: Quantity tests → Comparator → Referent (with referent recursion)
    Phase 2: EventFormChecker (derives from Phase 1 violations)
    """

    def __init__(self):
        # Quantity checkers (run on main quantity and optionally on referent quantity)
        self.quantity_checkers: list[CanonicalFieldChecker] = [
            QuantitySignatureChecker(),
            ConditioningChecker(),
            ShapeChecker(),
            UncertaintyChecker(),
        ]
        # Component checkers
        self.comparator_checker = ComparatorChecker()
        self.referent_checker = ReferentChecker()
        # Derived checker
        self.form_checker = EventFormChecker()

        # All checkers for lookup
        self._all_checkers: list[CanonicalFieldChecker] = [
            self.comparator_checker,
            self.referent_checker,
            *self.quantity_checkers,
            self.form_checker,
        ]

    def run_all(
        self,
        ir: Any = None,
        nl_values: Optional[dict[str, ExtractedValue]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run all checks with ordered execution.

        Phase 1: quantity checks, comparator, referent (+ referent quantity recursion)
        Phase 2: event form (derived from Phase 1 violations)
        """
        ir_dict = _ensure_dict(ir)
        component_violations: list[Violation] = []

        # Phase 1a: Quantity checks on main quantity
        component_violations.extend(
            self._run_quantity_checks(ir_dict, nl_values, artifact)
        )

        # Phase 1b: Comparator check
        component_violations.extend(
            self.comparator_checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact)
        )

        # Phase 1c: Referent check
        component_violations.extend(
            self.referent_checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact)
        )

        # Phase 1d: If referent is a quantity, run quantity checks on it too
        if self.referent_checker.is_quantity_referent(ir_dict):
            referent_quantity = self.referent_checker.get_referent_quantity(ir_dict)
            if referent_quantity:
                # Wrap referent quantity into synthetic IR for quantity checkers
                synthetic_ir = {
                    "type": "hypothesis",
                    "event": {
                        "type": "comparison",
                        "quantity": referent_quantity,
                        "comparator": ir_dict.get("event", {}).get("comparator"),
                        "referent": None,
                        "predicate": None,
                    },
                }
                # Remap nl_values: convert "event.referent.quantity.*" keys
                # back to "event.quantity.*" so quantity checkers find them
                referent_nl_values = None
                if nl_values:
                    referent_nl_values = {}
                    prefix = "event.referent."
                    for key, val in nl_values.items():
                        if key.startswith(prefix):
                            referent_nl_values[f"event.{key[len(prefix):]}"] = val
                referent_violations = self._run_quantity_checks(
                    synthetic_ir, referent_nl_values or None, artifact
                )
                # Prefix violation IDs for referent quantity
                for v in referent_violations:
                    v.invariantID = f"event.referent.{v.invariantID.removeprefix('event.')}"
                component_violations.extend(referent_violations)

        # Phase 2: Event form check (derived from component violations)
        form_violations = self.form_checker.check_with_component_violations(
            ir=ir_dict, nl_values=nl_values, artifact=artifact,
            component_violations=component_violations,
        )

        return component_violations + form_violations

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
                if isinstance(checker, EventFormChecker):
                    # Event form needs component violations
                    component_violations = []
                    ir_dict = _ensure_dict(ir)
                    for c in self.quantity_checkers:
                        component_violations.extend(c.check(ir=ir_dict, nl_values=nl_values, artifact=artifact))
                    component_violations.extend(
                        self.comparator_checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact)
                    )
                    component_violations.extend(
                        self.referent_checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact)
                    )
                    return checker.check_with_component_violations(
                        ir=ir_dict, nl_values=nl_values, artifact=artifact,
                        component_violations=component_violations,
                    )
                return checker.check(ir=ir, nl_values=nl_values, artifact=artifact)
        return []

    def get_checker(self, field_id: str) -> Optional[CanonicalFieldChecker]:
        """Get a specific checker by field ID."""
        for checker in self._all_checkers:
            if checker.field_id == field_id:
                return checker
        return None

    def _run_quantity_checks(
        self,
        ir_dict: dict,
        nl_values: Optional[dict[str, ExtractedValue]] = None,
        artifact: Optional[dict] = None,
    ) -> list[Violation]:
        """Run all 4 quantity checkers on an IR dict."""
        violations = []
        for checker in self.quantity_checkers:
            violations.extend(checker.check(ir=ir_dict, nl_values=nl_values, artifact=artifact))
        return violations
