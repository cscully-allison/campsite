"""Tests for the Campsite Python backend."""

import pytest
import json


class TestIRParser:
    """Tests for the hypothesis IR parser."""

    def test_parse_simple_expectation(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | y = 1] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["quantity"]["attr"] == "x"
        assert d["event"]["quantity"]["predicate"]["attr"] == "y"
        assert d["event"]["quantity"]["predicate"]["comparator"] == "="
        assert d["event"]["quantity"]["predicate"]["value"] == 1.0
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["value"] == 0.0

    def test_parse_contrast(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | g = 1] - E[x | g = 0] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "contrast"
        assert d["event"]["quantity"]["lhs"]["type"] == "expectation"
        assert d["event"]["quantity"]["rhs"]["type"] == "expectation"
        assert d["event"]["quantity"]["op"] == "-"

    def test_parse_rv(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("bootstrap(E[x | y = 1]) BETWEEN (0, 1)")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "rv"
        assert d["event"]["quantity"]["distribution"] == "bootstrap"
        assert d["event"]["quantity"]["estimand"]["type"] == "expectation"
        assert d["event"]["comparator"] == "BETWEEN"
        assert d["event"]["referent"]["value"] == [0.0, 1.0]

    def test_parse_string_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis('E[salary | dept = "Engineering"] > 50000')
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["quantity"]["predicate"]["value"] == "Engineering"
        assert d["event"]["referent"]["value"] == 50000.0

    def test_parse_boolean_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[failure | flag = true] / E[failure | flag = false] > 1")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "contrast"
        assert d["event"]["quantity"]["lhs"]["predicate"]["value"] is True
        assert d["event"]["quantity"]["rhs"]["predicate"]["value"] is False
        assert d["event"]["quantity"]["op"] == "/"

    def test_parse_quantity_vs_quantity(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] > E[x | a = 0]")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["quantity"]["attr"] == "x"
        assert d["event"]["quantity"]["predicate"]["value"] == 1.0
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["type"] == "expectation"
        assert d["event"]["referent"]["attr"] == "x"
        assert d["event"]["referent"]["predicate"]["value"] == 0.0

    def test_parse_conjunction_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1 ^ b = 2] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["predicate"]
        assert pred["type"] == "predicate"
        assert pred["kind"] == "conjunction"
        assert pred["lhs"]["attr"] == "a"
        assert pred["lhs"]["value"] == 1.0
        assert pred["rhs"]["attr"] == "b"
        assert pred["rhs"]["value"] == 2.0

    def test_parse_triple_conjunction_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1 ^ b = 2 ^ c = 3] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["predicate"]
        assert pred["kind"] == "conjunction"
        # Left-associative: (a=1 ^ b=2) ^ c=3
        assert pred["lhs"]["kind"] == "conjunction"
        assert pred["lhs"]["lhs"]["attr"] == "a"
        assert pred["lhs"]["rhs"]["attr"] == "b"
        assert pred["rhs"]["attr"] == "c"

    def test_parse_extract(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("Extract(linear_model, E[y | x = 1]) > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "extract"
        assert d["event"]["quantity"]["model"] == "linear_model"
        assert d["event"]["quantity"]["estimand"]["type"] == "expectation"
        assert d["event"]["quantity"]["estimand"]["attr"] == "y"

    def test_parse_event_with_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] > 5 (b = 2)")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["value"] == 5.0
        assert d["event"]["predicate"]["attr"] == "b"
        assert d["event"]["predicate"]["comparator"] == "="
        assert d["event"]["predicate"]["value"] == 2.0

    def test_parse_event_without_predicate_has_none(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] > 5")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["predicate"] is None

    def test_parse_thorn_unspecified_comparator(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] ᚦ 5")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == "ᚦ"
        assert d["event"]["referent"]["value"] == 5.0

    def test_parse_thorn_unspecified_referent(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] > ᚦ")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["type"] == "unspecified"

    def test_parse_thorn_unspecified_comparator_quantity_referent(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | a = 1] ᚦ E[x | a = 0]")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["comparator"] == "ᚦ"
        assert d["event"]["referent"]["type"] == "expectation"
        assert d["event"]["referent"]["attr"] == "x"

    def test_parse_error_returns_error_node(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("invalid hypothesis string without structure")
        assert parse_result.is_partial
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "error"
        assert "Parse error" in d["event"]["message"]


class TestSICheckers:
    """Tests for the canonical field semantic invariant checkers."""

    # -- IR Fixtures --

    SIMPLE_EXPECTATION_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "expectation",
                "attr": "x",
                "predicate": {"type": "predicate", "kind": "comparison", "attr": "y", "comparator": "=", "value": 1},
            },
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
            "predicate": None,
        },
    }

    CONTRAST_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "contrast",
                "lhs": {"type": "expectation", "attr": "x", "predicate": {"type": "predicate", "kind": "comparison", "attr": "g", "comparator": "=", "value": 1}},
                "op": "-",
                "rhs": {"type": "expectation", "attr": "x", "predicate": {"type": "predicate", "kind": "comparison", "attr": "g", "comparator": "=", "value": 0}},
            },
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
            "predicate": None,
        },
    }

    RV_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "rv",
                "distribution": "bootstrap",
                "estimand": {
                    "type": "expectation",
                    "attr": "x",
                    "predicate": {"type": "predicate", "kind": "comparison", "attr": "y", "comparator": "=", "value": 1},
                },
            },
            "comparator": "BETWEEN",
            "referent": {"type": "const", "value": [0, 1]},
            "predicate": None,
        },
    }

    UNDERSPECIFIED_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "attr": "x", "predicate": None},
            "comparator": "\u16A6",
            "referent": {"type": "const", "value": 5},
            "predicate": None,
        },
    }

    EMPTY_EVENT_IR = {"type": "hypothesis", "event": {}}

    CONDITIONED_EVENT_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "attr": "x", "predicate": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 1}},
            "comparator": ">",
            "referent": {"type": "const", "value": 5},
            "predicate": {"type": "predicate", "kind": "comparison", "attr": "b", "comparator": "=", "value": 2},
        },
    }

    QUANTITY_REFERENT_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "attr": "x", "predicate": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 1}},
            "comparator": ">",
            "referent": {"type": "expectation", "attr": "x", "predicate": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 0}},
            "predicate": None,
        },
    }

    # -- ComparatorChecker tests --

    def test_comparator_extracts_valid(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == ">"
        assert result.exists is True

    def test_comparator_missing_in_empty_event(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        result = checker.extract_from_ir(self.EMPTY_EVENT_IR)
        assert result.exists is False

    def test_comparator_invalid_not_in_valid_set(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        ir = {
            "type": "hypothesis",
            "event": {"type": "comparison", "quantity": {}, "comparator": "~~", "referent": {"value": 0}},
        }
        checker = ComparatorChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "~~"
        assert result.exists is False

    def test_comparator_thorn_is_missing(self):
        """Thorn character 'ᚦ' should be extracted as 'missing'."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        result = checker.extract_from_ir(self.UNDERSPECIFIED_IR)
        assert result.value == "missing"
        assert result.exists is True

    def test_comparator_existence_violation_on_missing(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        violations = checker.check(ir=self.EMPTY_EVENT_IR)
        ids = [v.invariantID for v in violations]
        assert "event.comparator-EX-IR" in ids

    def test_comparator_no_violation_on_valid(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR)
        ex_violations = [v for v in violations if "EX-IR" in v.invariantID and "comparator" in v.invariantID]
        assert len(ex_violations) == 0

    # -- ReferentChecker tests --

    def test_referent_extracts_threshold(self):
        """Const referent should be categorized as 'threshold'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "threshold"
        assert result.exists is True

    def test_referent_extracts_missing_for_unspecified(self):
        """Unspecified referent should be categorized as 'missing'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x"},
                "comparator": ">",
                "referent": {"type": "unspecified"},
            },
        }
        checker = ReferentChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "missing"
        assert result.exists is True
        assert result.metadata.get("unspecified") is True

    def test_referent_extracts_missing_for_none(self):
        """None referent should be categorized as 'missing'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x"},
                "comparator": ">",
                "referent": None,
            },
        }
        checker = ReferentChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "missing"
        assert result.exists is True

    def test_referent_extracts_quantity(self):
        """Quantity referent should be categorized as 'quantity'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_ir(self.QUANTITY_REFERENT_IR)
        assert result.value == "quantity"
        assert result.exists is True

    def test_referent_is_quantity_referent(self):
        """is_quantity_referent should detect quantity-type referents."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        assert checker.is_quantity_referent(self.QUANTITY_REFERENT_IR) is True
        assert checker.is_quantity_referent(self.SIMPLE_EXPECTATION_IR) is False

    def test_referent_get_referent_quantity(self):
        """get_referent_quantity should return the referent dict for quantity referents."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        ref_qty = checker.get_referent_quantity(self.QUANTITY_REFERENT_IR)
        assert ref_qty is not None
        assert ref_qty["type"] == "expectation"
        assert checker.get_referent_quantity(self.SIMPLE_EXPECTATION_IR) is None

    # -- EventFormChecker tests --

    def test_event_form_quantity_comp_threshold_conditioned(self):
        """Expectation with predicate + const referent = quantity_comp_threshold_conditioned."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        checker = EventFormChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "quantity_comp_threshold_conditioned"
        assert result.exists is True

    def test_event_form_quantity_comp_threshold(self):
        """No predicates + const referent = quantity_comp_threshold."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        checker = EventFormChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "quantity_comp_threshold"

    def test_event_form_quantity_comp_quantity(self):
        """No predicates + quantity referent = quantity_comp_quantity."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {"type": "expectation", "attr": "y", "predicate": None},
                "predicate": None,
            },
        }
        checker = EventFormChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "quantity_comp_quantity"

    def test_event_form_quantity_comp_quantity_conditioned(self):
        """Predicate on referent quantity = quantity_comp_quantity_conditioned."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        checker = EventFormChecker()
        result = checker.extract_from_ir(self.QUANTITY_REFERENT_IR)
        assert result.value == "quantity_comp_quantity_conditioned"

    def test_event_form_conditioned_event_predicate(self):
        """Event-level predicate should produce conditioned form."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        checker = EventFormChecker()
        result = checker.extract_from_ir(self.CONDITIONED_EVENT_IR)
        assert result.value == "quantity_comp_threshold_conditioned"

    def test_event_form_empty_event(self):
        """Empty event should return no value."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        checker = EventFormChecker()
        result = checker.extract_from_ir(self.EMPTY_EVENT_IR)
        assert result.exists is False

    def test_event_form_compromised_by_component_violations(self):
        """EventFormChecker should report compromised form when component checks fail."""
        from campsite.campsite_lib.si_checkers import EventFormChecker, Violation, ViolationType, Criticality

        checker = EventFormChecker()
        component_violations = [
            Violation(
                invariantID="event.comparator-EX-IR",
                violationType=ViolationType.MISSING_IN_IR,
                message="missing comparator",
                criticality=Criticality.WARN,
            ),
        ]
        violations = checker.check_with_component_violations(
            ir=self.SIMPLE_EXPECTATION_IR,
            component_violations=component_violations,
        )
        compromised = [v for v in violations if "COMPROMISED" in v.invariantID]
        assert len(compromised) == 1
        assert "comparator" in compromised[0].message

    # -- QuantitySignatureChecker tests --

    def test_quantity_signature_level(self):
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "level"

    def test_quantity_signature_contrast(self):
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_ir(self.CONTRAST_IR)
        assert result.value == "contrast"

    def test_quantity_signature_distribution(self):
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_ir(self.RV_IR)
        assert result.value == "distribution"

    def test_quantity_signature_unknown(self):
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_ir(self.EMPTY_EVENT_IR)
        assert result.exists is False

    # -- ConditioningChecker tests --

    def test_conditioning_extracts_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditioningChecker

        checker = ConditioningChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.exists is True
        assert len(result.value) == 1
        assert result.value[0]["attr"] == "y"

    def test_conditioning_contrast_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditioningChecker

        checker = ConditioningChecker()
        result = checker.extract_from_ir(self.CONTRAST_IR)
        assert result.exists is True
        assert len(result.value) == 2

    def test_conditioning_no_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditioningChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        checker = ConditioningChecker()
        result = checker.extract_from_ir(ir)
        assert result.exists is False
        assert result.value == []

    def test_conditioning_no_existence_violation(self):
        """Conditioning is optional — absence should not generate a violation."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        checker = ConditioningChecker()
        violations = checker.check(ir=ir)
        ex_violations = [v for v in violations if "EX-" in v.invariantID]
        assert len(ex_violations) == 0

    def test_conditioning_includes_event_predicate(self):
        from campsite.campsite_lib.si_checkers import ConditioningChecker

        checker = ConditioningChecker()
        result = checker.extract_from_ir(self.CONDITIONED_EVENT_IR)
        attrs = [p["attr"] for p in result.value]
        assert "a" in attrs
        assert "b" in attrs

    # -- ShapeChecker tests --

    def test_shape_value(self):
        """Expectation should have shape 'value'."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        checker = ShapeChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "value"

    def test_shape_difference(self):
        """Contrast with '-' should have shape 'difference'."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        checker = ShapeChecker()
        result = checker.extract_from_ir(self.CONTRAST_IR)
        assert result.value == "difference"

    def test_shape_ratio(self):
        """Contrast with '/' should have shape 'ratio'."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {
                    "type": "contrast",
                    "lhs": {"type": "expectation", "attr": "x"},
                    "op": "/",
                    "rhs": {"type": "expectation", "attr": "x"},
                },
                "comparator": ">",
                "referent": {"type": "const", "value": 1},
            },
        }
        checker = ShapeChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "ratio"

    def test_shape_nested_flattened_to_difference(self):
        """RV wrapping a contrast should flatten to 'difference' (not 'nested_difference')."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {
                    "type": "rv",
                    "distribution": "bootstrap",
                    "estimand": {
                        "type": "contrast",
                        "lhs": {"type": "expectation", "attr": "x"},
                        "op": "-",
                        "rhs": {"type": "expectation", "attr": "x"},
                    },
                },
                "comparator": "BETWEEN",
                "referent": {"type": "const", "value": [0, 1]},
            },
        }
        checker = ShapeChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "difference"

    # -- UncertaintyChecker tests (collapsed from shown + target) --

    def test_uncertainty_missing(self):
        """Non-RV quantity should have uncertainty 'missing'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker

        checker = UncertaintyChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "missing"

    def test_uncertainty_attached(self):
        """RV quantity should have uncertainty 'attached'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker

        checker = UncertaintyChecker()
        result = checker.extract_from_ir(self.RV_IR)
        assert result.value == "attached"

    # -- Pairwise check tests (using nl_values dict) --

    def test_pairwise_comparator_match(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue

        checker = ComparatorChecker()
        nl_values = {"event.comparator": ExtractedValue(value=">", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_comparator_mismatch(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue

        checker = ComparatorChecker()
        nl_values = {"event.comparator": ExtractedValue(value="<", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1
        assert pw_violations[0].expected == "<"
        assert pw_violations[0].observed == ">"

    def test_pairwise_skipped_when_source_missing(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue

        checker = ComparatorChecker()
        nl_values = {"event.comparator": ExtractedValue(value=None, exists=False)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-" in v.invariantID]
        assert len(pw_violations) == 0

    # -- Pairwise with nl_values dict --

    def test_pairwise_referent_match(self):
        """NL 'threshold' should match IR 'threshold'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker, ExtractedValue

        checker = ReferentChecker()
        nl_values = {"event.referent": ExtractedValue(value="threshold", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_referent_mismatch(self):
        """NL 'quantity' should NOT match IR 'threshold'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker, ExtractedValue

        checker = ReferentChecker()
        nl_values = {"event.referent": ExtractedValue(value="quantity", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    def test_pairwise_shape_match(self):
        """NL 'difference' should match IR 'difference'."""
        from campsite.campsite_lib.si_checkers import ShapeChecker, ExtractedValue

        checker = ShapeChecker()
        nl_values = {"event.quantity.shape": ExtractedValue(value="difference", exists=True)}
        violations = checker.check(ir=self.CONTRAST_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_uncertainty_match(self):
        """NL 'attached' should match IR 'attached'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker, ExtractedValue

        checker = UncertaintyChecker()
        nl_values = {"event.quantity.uncertainty": ExtractedValue(value="attached", exists=True)}
        violations = checker.check(ir=self.RV_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_uncertainty_mismatch(self):
        """NL 'attached' should NOT match IR 'missing'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker, ExtractedValue

        checker = UncertaintyChecker()
        nl_values = {"event.quantity.uncertainty": ExtractedValue(value="attached", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    # ConditioningChecker — fast path (both dicts)

    def test_pairwise_conditioning_both_dicts_match(self):
        """Two dict lists with same attrs in different order should match."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker, ExtractedValue

        checker = ConditioningChecker()
        nl_values = {"event.quantity.conditioning": ExtractedValue(
            value=[{"attr": "b", "comparator": "=", "value": 2}, {"attr": "a", "comparator": "=", "value": 1}],
            exists=True,
        )}
        violations = checker.check(ir=self.CONDITIONED_EVENT_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_conditioning_both_dicts_mismatch(self):
        """Two dict lists with different attrs should NOT match."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker, ExtractedValue

        checker = ConditioningChecker()
        nl_values = {"event.quantity.conditioning": ExtractedValue(
            value=[{"attr": "z", "comparator": "=", "value": 9}],
            exists=True,
        )}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    # ConditioningChecker — count mismatch (no LLM needed)

    def test_pairwise_conditioning_count_mismatch(self):
        """Different number of conditions should NOT match without LLM."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker, ExtractedValue

        checker = ConditioningChecker()
        nl_values = {"event.quantity.conditioning": ExtractedValue(
            value=["y = 1"],
            exists=True,
        )}
        # CONDITIONED_EVENT_IR has 2 predicates (a and b)
        violations = checker.check(ir=self.CONDITIONED_EVENT_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    # ConditioningChecker — LLM judge path (using override)

    def test_pairwise_conditioning_judge_override_match(self):
        """Mixed types with judge override True should match."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker, ExtractedValue

        checker = ConditioningChecker()
        checker.set_judge_override(True)
        nl_values = {"event.quantity.conditioning": ExtractedValue(
            value=["for jobs that request GPUs", "for jobs with low GPU utilization", "for high-GPU-utilization jobs"],
            exists=True,
        )}
        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {
                    "type": "expectation", "attr": "x",
                    "predicate": {
                        "type": "predicate", "kind": "conjunction",
                        "lhs": {
                            "type": "predicate", "kind": "conjunction",
                            "lhs": {"type": "predicate", "kind": "comparison", "attr": "gpu", "comparator": "=", "value": True},
                            "rhs": {"type": "predicate", "kind": "comparison", "attr": "utilization", "comparator": "<", "value": 0.5},
                        },
                        "rhs": {"type": "predicate", "kind": "comparison", "attr": "utilization", "comparator": ">", "value": 0.8},
                    },
                },
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        violations = checker.check(ir=ir, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_conditioning_judge_override_no_match(self):
        """Mixed types with judge override False should NOT match."""
        from campsite.campsite_lib.si_checkers import ConditioningChecker, ExtractedValue

        checker = ConditioningChecker()
        checker.set_judge_override(False)
        nl_values = {"event.quantity.conditioning": ExtractedValue(
            value=["for jobs that request GPUs", "for jobs with low GPU utilization", "for high-GPU-utilization jobs"],
            exists=True,
        )}
        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {
                    "type": "expectation", "attr": "x",
                    "predicate": {
                        "type": "predicate", "kind": "conjunction",
                        "lhs": {
                            "type": "predicate", "kind": "conjunction",
                            "lhs": {"type": "predicate", "kind": "comparison", "attr": "gpu", "comparator": "=", "value": True},
                            "rhs": {"type": "predicate", "kind": "comparison", "attr": "utilization", "comparator": "<", "value": 0.5},
                        },
                        "rhs": {"type": "predicate", "kind": "comparison", "attr": "utilization", "comparator": ">", "value": 0.8},
                    },
                },
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        violations = checker.check(ir=ir, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    # -- Confidence threshold tests --

    def test_low_confidence_downgrades_to_warn(self):
        """Low-confidence NL extraction should produce WARN, not FAIL, on mismatch."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, Criticality

        checker = ComparatorChecker()
        nl_values = {"event.comparator": ExtractedValue(value="<", exists=True, confidence=0.2)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1
        assert pw_violations[0].criticality == Criticality.WARN

    def test_high_confidence_stays_fail(self):
        """High-confidence NL extraction should keep FAIL on mismatch."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, Criticality

        checker = ComparatorChecker()
        nl_values = {"event.comparator": ExtractedValue(value="<", exists=True, confidence=0.9)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1
        assert pw_violations[0].criticality == Criticality.FAIL

    # -- SICheckRunner tests --

    def test_runner_ir_only_no_violations_on_valid(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=self.SIMPLE_EXPECTATION_IR)
        # Valid IR should have no existence violations for required fields
        ex_ir_violations = [v for v in violations if "EX-IR" in v.invariantID]
        assert len(ex_ir_violations) == 0

    def test_runner_ir_only_violations_on_empty(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=self.EMPTY_EVENT_IR)
        # Should have multiple existence violations
        assert len(violations) > 0
        ids = [v.invariantID for v in violations]
        assert "event.comparator-EX-IR" in ids

    def test_runner_run_field(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_field("event.comparator", ir=self.SIMPLE_EXPECTATION_IR)
        # Should only return violations for that field
        for v in violations:
            assert "event.comparator" in v.invariantID

    def test_runner_get_checker(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner, ComparatorChecker

        runner = SICheckRunner()
        checker = runner.get_checker("event.comparator")
        assert isinstance(checker, ComparatorChecker)

    def test_runner_ordered_execution(self):
        """Runner should execute quantity/comparator/referent before form."""
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=self.EMPTY_EVENT_IR)
        ids = [v.invariantID for v in violations]
        # Form checker should have produced a COMPROMISED violation
        # since component checkers found violations
        compromised = [v for v in violations if "COMPROMISED" in v.invariantID]
        assert len(compromised) == 1

    def test_runner_referent_quantity_recursion(self):
        """When referent is a quantity, quantity checks should run on it with prefixed IDs."""
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=self.QUANTITY_REFERENT_IR)
        ids = [v.invariantID for v in violations]
        # Check that referent quantity checks produced prefixed violations
        # The referent is a valid expectation, so no EX violations expected,
        # but we should see no errors either
        referent_violations = [v for v in violations if "event.referent." in v.invariantID]
        # Referent quantity is valid — should have no violations
        assert all("EX-IR" not in v.invariantID for v in referent_violations)

    def test_runner_with_nl_values(self):
        """Runner should accept nl_values dict for pairwise checking."""
        from campsite.campsite_lib.si_checkers import SICheckRunner, ExtractedValue

        runner = SICheckRunner()
        nl_values = {
            "event.comparator": ExtractedValue(value="<", exists=True),
        }
        violations = runner.run_all(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID and "comparator" in v.invariantID]
        assert len(pw_violations) == 1

    # -- Violation serialization --

    def test_violation_to_dict(self):
        from campsite.campsite_lib.si_checkers import Violation, ViolationType, Criticality

        v = Violation(
            invariantID="event.comparator-EX-IR",
            violationType=ViolationType.MISSING_IN_IR,
            message="test",
            criticality=Criticality.WARN,
            expected="present",
            observed=None,
        )
        d = v.to_dict()
        assert d["invariantID"] == "event.comparator-EX-IR"
        assert d["violationType"] == "missing_in_ir"
        assert d["criticality"] == "warn"

    def test_violation_type_nl_artifact(self):
        """NL_ARTIFACT_MISMATCH violation type should exist."""
        from campsite.campsite_lib.si_checkers import ViolationType

        assert ViolationType.NL_ARTIFACT_MISMATCH.value == "nl_artifact_mismatch"

    # -- Integration test: parser + runner --

    def test_parsed_hypothesis_through_runner(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict
        from campsite.campsite_lib.si_checkers import SICheckRunner

        parse_result = parse_hypothesis("E[x | y = 1] > 0")
        assert parse_result.errors == []
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)
        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=ir_dict)
        ex_ir_violations = [v for v in violations if "EX-IR" in v.invariantID]
        assert len(ex_ir_violations) == 0

    def test_parsed_contrast_through_runner(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict
        from campsite.campsite_lib.si_checkers import SICheckRunner

        parse_result = parse_hypothesis("E[x | g = 1] - E[x | g = 0] > 0")
        assert parse_result.errors == []
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)
        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=ir_dict)
        ex_ir_violations = [v for v in violations if "EX-IR" in v.invariantID]
        assert len(ex_ir_violations) == 0

    # -- EventFormChecker referent predicate --

    def test_event_form_conditioned_by_referent_predicate(self):
        """EventFormChecker should detect conditioning in referent subtree."""
        from campsite.campsite_lib.si_checkers import EventFormChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {
                    "type": "expectation", "attr": "x",
                    "predicate": {"type": "predicate", "kind": "comparison", "attr": "g", "comparator": "=", "value": 0},
                },
                "predicate": None,
            },
        }
        checker = EventFormChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == "quantity_comp_quantity_conditioned"

    # -- ErrorNode produces MALFORMED violations --

    def test_error_node_event_produces_malformed(self):
        """ErrorNode at event level should produce MALFORMED violations."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ViolationType

        ir = {
            "type": "hypothesis",
            "event": {"type": "error", "boundary": "event", "message": "bad parse"},
        }
        checker = ComparatorChecker()
        violations = checker.check(ir=ir)
        malformed = [v for v in violations if v.violationType == ViolationType.MALFORMED]
        assert len(malformed) == 1
        assert "malformed" in malformed[0].message.lower()
        assert "event" in malformed[0].message.lower()

    def test_error_node_quantity_only_affects_quantity_checkers(self):
        """ErrorNode in quantity should not affect ComparatorChecker."""
        from campsite.campsite_lib.si_checkers import (
            ComparatorChecker, QuantitySignatureChecker, ViolationType,
        )

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "error", "boundary": "quantity", "message": "bad qty"},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
                "predicate": None,
            },
        }
        # ComparatorChecker should work fine (event is not an error)
        comp_checker = ComparatorChecker()
        comp_violations = comp_checker.check(ir=ir)
        comp_malformed = [v for v in comp_violations if v.violationType == ViolationType.MALFORMED]
        assert len(comp_malformed) == 0

        # QuantitySignatureChecker should report MALFORMED
        qty_checker = QuantitySignatureChecker()
        qty_violations = qty_checker.check(ir=ir)
        qty_malformed = [v for v in qty_violations if v.violationType == ViolationType.MALFORMED]
        assert len(qty_malformed) == 1

    def test_error_node_referent_affects_referent_checker(self):
        """ErrorNode in referent should produce MALFORMED from ReferentChecker."""
        from campsite.campsite_lib.si_checkers import ReferentChecker, ViolationType

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "attr": "x", "predicate": None},
                "comparator": ">",
                "referent": {"type": "error", "boundary": "referent", "message": "bad ref"},
                "predicate": None,
            },
        }
        checker = ReferentChecker()
        violations = checker.check(ir=ir)
        malformed = [v for v in violations if v.violationType == ViolationType.MALFORMED]
        assert len(malformed) == 1


class TestComparatorScanner:
    """Tests for the bracket-aware comparator scanner."""

    def test_simple_greater_than(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("x > 5")
        assert result is not None
        assert result[0] == ">"

    def test_skips_bracket_depth(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("E[x | y = 1] > 5")
        assert result is not None
        comp, start, end = result
        assert comp == ">"
        # ">" should be after the "]", not the "=" inside brackets
        assert start > 12

    def test_longest_match_first(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("x >= 5")
        assert result is not None
        assert result[0] == ">="

    def test_between(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("x BETWEEN (0, 1)")
        assert result is not None
        assert result[0] == "BETWEEN"

    def test_no_comparator(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("just some text")
        assert result is None

    def test_nested_parens(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("bootstrap(E[x | y = 1]) BETWEEN (0, 1)")
        assert result is not None
        assert result[0] == "BETWEEN"

    def test_thorn(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("E[x | a = 1] \u16A6 5")
        assert result is not None
        assert result[0] == "\u16A6"

    def test_contrast_inner_ops_skipped(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("E[x | g = 1] - E[x | g = 0] > 0")
        assert result is not None
        assert result[0] == ">"


class TestPartialRecovery:
    """Tests for partial parse recovery."""

    def test_recovery_bad_quantity_good_referent(self):
        """Malformed quantity but valid comparator and referent."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        # "E[x]" is not a valid expectation (missing "| predicate"); "> 5" is valid
        parse_result = parse_hypothesis("E[x] > 5")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["quantity"]["type"] == "error"
        assert d["event"]["quantity"]["boundary"] == "quantity"
        assert d["event"]["referent"]["value"] == 5.0

    def test_recovery_good_quantity_bad_referent(self):
        """Valid quantity and comparator but malformed referent."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | y = 1] > E[z |")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["quantity"]["attr"] == "x"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["type"] == "error"
        assert d["event"]["referent"]["boundary"] == "referent"

    def test_recovery_both_sides_bad(self):
        """Both quantity and referent are malformed, but comparator found."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("$$$ > ???")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert len(parse_result.errors) == 2
        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["quantity"]["type"] == "error"
        assert d["event"]["referent"]["type"] == "error"

    def test_recovery_no_comparator_falls_back(self):
        """No comparator at depth 0 — falls back to full event ErrorNode."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("just random words")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "error"
        assert d["event"]["boundary"] == "event"

    def test_recovery_missing_referent(self):
        """Comparator found but nothing after — quantity OK, referent error."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | y = 1] >")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["type"] == "error"

    def test_recovery_between_comparator(self):
        """Recovery works with multi-character comparators like BETWEEN."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("$$$ BETWEEN (0, 1)")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == "BETWEEN"
        assert d["event"]["quantity"]["type"] == "error"

    def test_recovery_thorn_comparator(self):
        """Recovery works with thorn comparator."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("$$$ \u16A6 5")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == "\u16A6"
        assert d["event"]["quantity"]["type"] == "error"

    def test_recovery_errors_have_positions(self):
        """ErrorNodes from recovery should have start/end positions."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis

        parse_result = parse_hypothesis("$$$ > ???")
        for err in parse_result.errors:
            assert isinstance(err.start, int)
            assert isinstance(err.end, int)

    def test_recovery_preserves_quantity_with_bad_referent(self):
        """When quantity parses but referent fails, quantity is preserved."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x | y = 1] >= E[z |")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == ">="
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["referent"]["type"] == "error"

    def test_successful_parse_returns_empty_errors(self):
        """Successful full parse returns ParseResult with empty errors."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis

        parse_result = parse_hypothesis("E[x | y = 1] > 0")
        assert not parse_result.is_partial
        assert parse_result.errors == []
        assert parse_result.hypothesis.type == "hypothesis"
        assert parse_result.hypothesis.event.type == "comparison"

    def test_parse_result_to_dict(self):
        """parse_result_to_dict serializes both hypothesis and errors."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, parse_result_to_dict

        parse_result = parse_hypothesis("$$$ > 5")
        d = parse_result_to_dict(parse_result)

        assert "hypothesis" in d
        assert "errors" in d
        assert isinstance(d["errors"], list)
        assert len(d["errors"]) > 0


class TestAnalysisState:
    """Tests for the AnalysisState model."""

    def test_default_state(self):
        from campsite.campsite_lib.utils import AnalysisState

        state = AnalysisState()

        assert state.stage == "conversation"
        assert state.initialUserQuestion == ""
        assert state.clarifications == []
        assert state.turns == 0
        assert state.waitingForUser == False
        assert state.substage == "none"

    def test_state_with_values(self):
        from campsite.campsite_lib.utils import AnalysisState

        state = AnalysisState(
            stage="hypothesis",
            initialUserQuestion="Is X > Y?",
            clarifications=["Yes", "No"],
            turns=3,
        )

        assert state.stage == "hypothesis"
        assert state.initialUserQuestion == "Is X > Y?"
        assert len(state.clarifications) == 2
        assert state.turns == 3

    def test_state_serialization(self):
        from campsite.campsite_lib.utils import AnalysisState

        state = AnalysisState(
            stage="artifactgen",
            hypothesis="E[x] > 0",
        )

        # Should serialize to JSON
        json_str = state.model_dump_json()
        data = json.loads(json_str)

        assert data["stage"] == "artifactgen"
        assert data["hypothesis"] == "E[x] > 0"


class TestContracts:
    """Tests for artifact contracts."""

    def test_code_artifact_contract(self):
        from campsite.campsite_lib.contracts import code_artifact_contract

        assert code_artifact_contract["uncertainty_estimator"] == "bootstrap"
        assert code_artifact_contract["n_draws"] == 1000

    def test_vis_artifact_contract(self):
        from campsite.campsite_lib.contracts import vis_artifact_contract

        assert vis_artifact_contract["output_format"] == "altair"
        assert "x" in vis_artifact_contract["encodings"]
        assert "y" in vis_artifact_contract["encodings"]
        assert "point" in vis_artifact_contract["continuous_markings"]


class TestIRAst:
    """Tests for IR AST dataclasses."""

    def test_comparison_const_referent_dataclass(self):
        from campsite.campsite_lib.ir_ast import Hypothesis, Comparison, Expectation, Predicate, Const

        pred = Predicate(type="predicate", kind="comparison", attr="y", comparator="=", value=1)
        exp = Expectation(type="expectation", attr="x", predicate=pred)
        ref = Const(type="const", value=0)
        comp = Comparison(type="comparison", quantity=exp, comparator=">", referent=ref)
        hyp = Hypothesis(type="hypothesis", event=comp)

        assert hyp.type == "hypothesis"
        assert hyp.event.type == "comparison"
        assert hyp.event.quantity.attr == "x"
        assert hyp.event.referent.value == 0

    def test_comparison_quantity_referent_dataclass(self):
        from campsite.campsite_lib.ir_ast import Hypothesis, Comparison, Expectation, Predicate

        pred1 = Predicate(type="predicate", kind="comparison", attr="g", comparator="=", value=1)
        pred2 = Predicate(type="predicate", kind="comparison", attr="g", comparator="=", value=0)
        lhs = Expectation(type="expectation", attr="x", predicate=pred1)
        rhs = Expectation(type="expectation", attr="x", predicate=pred2)
        comp = Comparison(type="comparison", quantity=lhs, comparator=">", referent=rhs)
        hyp = Hypothesis(type="hypothesis", event=comp)

        assert hyp.type == "hypothesis"
        assert hyp.event.type == "comparison"
        assert hyp.event.quantity.attr == "x"
        assert hyp.event.referent.attr == "x"

    def test_comparison_with_predicate_dataclass(self):
        from campsite.campsite_lib.ir_ast import Comparison, Expectation, Predicate, Const

        pred = Predicate(type="predicate", kind="comparison", attr="a", comparator="=", value=1)
        exp = Expectation(type="expectation", attr="x", predicate=pred)
        event_pred = Predicate(type="predicate", kind="comparison", attr="b", comparator="=", value=2)
        comp = Comparison(type="comparison", quantity=exp, comparator=">", referent=Const(value=0), predicate=event_pred)

        assert comp.predicate.attr == "b"
        assert comp.predicate.value == 2

    def test_unspecified_dataclass(self):
        from campsite.campsite_lib.ir_ast import Comparison, Expectation, Predicate, Unspecified

        pred = Predicate(type="predicate", kind="comparison", attr="a", comparator="=", value=1)
        exp = Expectation(type="expectation", attr="x", predicate=pred)
        comp = Comparison(type="comparison", quantity=exp, comparator=">", referent=Unspecified())

        assert comp.referent.type == "unspecified"

    def test_conjunction_predicate_dataclass(self):
        from campsite.campsite_lib.ir_ast import Predicate

        p1 = Predicate(type="predicate", kind="comparison", attr="a", comparator="=", value=1)
        p2 = Predicate(type="predicate", kind="comparison", attr="b", comparator="=", value=2)
        conj = Predicate(type="predicate", kind="conjunction", lhs=p1, rhs=p2)

        assert conj.kind == "conjunction"
        assert conj.lhs.attr == "a"
        assert conj.rhs.attr == "b"

    def test_error_node(self):
        from campsite.campsite_lib.ir_ast import ErrorNode

        err = ErrorNode(
            type="error",
            boundary="event",
            message="Test error",
            text="bad input",
            start=0,
            end=9,
        )

        assert err.type == "error"
        assert err.boundary == "event"
        assert err.message == "Test error"


class TestNLExtractors:
    """Tests for the NL extractor classes."""

    def test_comparator_normalize_greater_than(self):
        from campsite.campsite_lib.nl_extractors import ComparatorNLExtractor

        extractor = ComparatorNLExtractor()
        assert extractor.normalize("greater than") == ">"
        assert extractor.normalize("higher than") == ">"
        assert extractor.normalize("no more than") == "<="
        assert extractor.normalize("between") == "BETWEEN"

    def test_signature_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import SignatureNLExtractor

        extractor = SignatureNLExtractor()
        assert extractor.normalize("mean") == "level"
        assert extractor.normalize("correlation") == "association"
        assert extractor.normalize("slope") == "trend"

    def test_shape_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import ShapeNLExtractor

        extractor = ShapeNLExtractor()
        assert extractor.normalize("subtraction") == "difference"
        assert extractor.normalize("division") == "ratio"
        assert extractor.normalize("simple") == "value"

    def test_uncertainty_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import UncertaintyNLExtractor

        extractor = UncertaintyNLExtractor()
        assert extractor.normalize("confidence interval") == "attached"
        assert extractor.normalize("none") == "missing"
        assert extractor.normalize("point") == "missing"

    def test_form_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import FormNLExtractor

        extractor = FormNLExtractor()
        assert extractor.normalize("conditional") == "quantity_comp_threshold_conditioned"
        assert extractor.normalize("unconditioned") == "quantity_comp_threshold"
        assert extractor.normalize("simple") == "quantity_comp_threshold"

    def test_nl_extractor_orchestrator(self):
        """NLExtractor.extract_all should return a dict with all field IDs."""
        from campsite.campsite_lib.nl_extractors import NLExtractor

        extractor = NLExtractor()
        # Can't actually call extract_all without LLM, but verify structure
        assert extractor.get_extractor("event.comparator") is not None
        assert extractor.get_extractor("event.referent") is not None
        assert extractor.get_extractor("event.quantity.signature") is not None
        assert extractor.get_extractor("event.quantity.conditioning") is not None
        assert extractor.get_extractor("event.quantity.shape") is not None
        assert extractor.get_extractor("event.quantity.uncertainty") is not None
        assert extractor.get_extractor("event.form") is not None
        assert extractor.get_extractor("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
