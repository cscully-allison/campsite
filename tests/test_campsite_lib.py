"""Tests for the Campsite Python backend."""

import pytest
import json


class TestIRParser:
    """Tests for the hypothesis IR parser."""

    def test_parse_simple_expectation(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (y = 1)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        qty_expr = d["event"]["quantity"]["expr"]
        assert qty_expr["type"] == "conditioned_expr"
        assert qty_expr["expr"]["type"] == "attr"
        assert qty_expr["expr"]["name"] == "x"
        assert qty_expr["predicate"]["attr"] == "y"
        assert qty_expr["predicate"]["comparator"] == "="
        assert qty_expr["predicate"]["value"] == 1.0
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["value"] == 0.0

    def test_parse_contrast(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (g = 1)] - E[x (g = 0)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "contrast"
        assert d["event"]["quantity"]["lhs"]["type"] == "expectation"
        assert d["event"]["quantity"]["rhs"]["type"] == "expectation"
        assert d["event"]["quantity"]["op"] == "-"

    def test_parse_rv(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("bootstrap(E[x (y = 1)]) BETWEEN (0, 1)")
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

        parse_result = parse_hypothesis('E[salary (dept = "Engineering")] > 50000')
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        qty_expr = d["event"]["quantity"]["expr"]
        assert qty_expr["predicate"]["value"] == "Engineering"
        assert d["event"]["referent"]["value"] == 50000.0

    def test_parse_boolean_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[failure (flag = true)] / E[failure (flag = false)] > 1")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "contrast"
        lhs_expr = d["event"]["quantity"]["lhs"]["expr"]
        rhs_expr = d["event"]["quantity"]["rhs"]["expr"]
        assert lhs_expr["predicate"]["value"] is True
        assert rhs_expr["predicate"]["value"] is False
        assert d["event"]["quantity"]["op"] == "/"

    def test_parse_conjunction_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (a = 1 ^ b = 2)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["type"] == "predicate"
        assert pred["kind"] == "conjunction"
        assert pred["lhs"]["attr"] == "a"
        assert pred["lhs"]["value"] == 1.0
        assert pred["rhs"]["attr"] == "b"
        assert pred["rhs"]["value"] == 2.0

    def test_parse_triple_conjunction_predicate(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (a = 1 ^ b = 2 ^ c = 3)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["kind"] == "conjunction"
        # Left-associative: (a=1 ^ b=2) ^ c=3
        assert pred["lhs"]["kind"] == "conjunction"
        assert pred["lhs"]["lhs"]["attr"] == "a"
        assert pred["lhs"]["rhs"]["attr"] == "b"
        assert pred["rhs"]["attr"] == "c"

    def test_parse_disjunction_predicate(self):
        """Disjunction predicate: E[x (a = 1 v b = 2)] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (a = 1 v b = 2)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["type"] == "predicate"
        assert pred["kind"] == "disjunction"
        assert pred["lhs"]["attr"] == "a"
        assert pred["lhs"]["value"] == 1.0
        assert pred["rhs"]["attr"] == "b"
        assert pred["rhs"]["value"] == 2.0

    def test_parse_attr_var_predicate(self):
        """Attribute variable as predicate value: E[x (threshold = cutoff)] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (threshold = cutoff)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["attr"] == "threshold"
        assert pred["comparator"] == "="
        assert pred["value"]["type"] == "attr"
        assert pred["value"]["name"] == "cutoff"

    def test_parse_func_predicate(self):
        """Function call as predicate value: E[x (salary > AVG(salary))] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (salary > AVG(salary))] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["attr"] == "salary"
        assert pred["comparator"] == ">"
        assert pred["value"]["type"] == "func"
        assert pred["value"]["name"] == "AVG"

    def test_parse_binary_predicate_value(self):
        """Binary expression as predicate value: E[x (val > baseline + 10)] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (val > baseline + 10)] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["attr"] == "val"
        assert pred["comparator"] == ">"
        assert pred["value"]["type"] == "binary"
        assert pred["value"]["lhs"]["type"] == "attr"
        assert pred["value"]["lhs"]["name"] == "baseline"
        assert pred["value"]["op"] == "+"
        assert pred["value"]["rhs"] == 10.0

    def test_parse_extract_in_predicate(self):
        """Extract as predicate value: E[x (coeff > Extract(model, y))] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (coeff > Extract(model, y))] > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        pred = d["event"]["quantity"]["expr"]["predicate"]
        assert pred["attr"] == "coeff"
        assert pred["comparator"] == ">"
        assert pred["value"]["type"] == "extract"
        assert pred["value"]["model"] == "model"

    def test_parse_extract(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("Extract(linear_model, y (x = 1)) > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["quantity"]["type"] == "extract"
        assert d["event"]["quantity"]["model"] == "linear_model"
        assert d["event"]["quantity"]["expr"]["type"] == "conditioned_expr"
        assert d["event"]["quantity"]["expr"]["expr"]["name"] == "y"

    def test_parse_expr_with_predicate_func(self):
        """func_call with expr-level predicate: AVG(salary) (dept = "eng")."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis('AVG(salary) (dept = "eng") > 50000')
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        qty = d["event"]["quantity"]
        assert qty["type"] == "conditioned_expr"
        assert qty["expr"]["type"] == "func"
        assert qty["expr"]["name"] == "AVG"
        assert qty["predicate"]["attr"] == "dept"
        assert qty["predicate"]["comparator"] == "="
        assert qty["predicate"]["value"] == "eng"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["value"] == 50000.0

    def test_parse_expr_with_predicate_var(self):
        """attr var with expr-level predicate: salary (age > 30)."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary (age > 30) > 50000")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        qty = d["event"]["quantity"]
        assert qty["type"] == "conditioned_expr"
        assert qty["expr"]["type"] == "attr"
        assert qty["expr"]["name"] == "salary"
        assert qty["predicate"]["attr"] == "age"
        assert qty["predicate"]["comparator"] == ">"
        assert qty["predicate"]["value"] == 30.0

    def test_parse_expr_without_predicate_regression(self):
        """Plain expr without predicate should still work."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > 50000")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "attr"
        assert d["event"]["quantity"]["name"] == "salary"

    def test_parse_thorn_unspecified_comparator(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (a = 1)] \u16A6 5")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == "\u16A6"
        assert d["event"]["referent"]["value"] == 5.0

    def test_parse_thorn_unspecified_referent(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (a = 1)] > \u16A6")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["type"] == "unspecified"

    def test_parse_error_returns_error_node(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("invalid hypothesis string without structure")
        assert parse_result.is_partial
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "error"
        assert "Parse error" in d["event"]["message"]

    # -- New grammar feature tests --

    def test_parse_simple_expectation_no_predicate(self):
        """E[x] without predicate is now valid syntax."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x] > 5")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["quantity"]["expr"]["type"] == "attr"
        assert d["event"]["quantity"]["expr"]["name"] == "x"
        assert d["event"]["comparator"] == ">"
        assert d["event"]["referent"]["value"] == 5.0

    def test_parse_hypothesis_across(self):
        """Hypothesis with ACROSS partition."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > 50000 ACROSS department")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["event"]["type"] == "comparison"
        assert d["across_partition"]["type"] == "partition"
        assert d["across_partition"]["attr"] == "department"
        assert d["within_partition"] is None

    def test_parse_hypothesis_within(self):
        """Hypothesis with WITHIN partition."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[salary] > 50000 WITHIN region")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["within_partition"]["type"] == "partition"
        assert d["within_partition"]["attr"] == "region"
        assert d["across_partition"] is None

    def test_parse_hypothesis_across_within(self):
        """Hypothesis with ACROSS and WITHIN partitions."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > 50000 ACROSS dept WITHIN region")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["type"] == "hypothesis"
        assert d["across_partition"]["type"] == "partition"
        assert d["across_partition"]["attr"] == "dept"
        assert d["within_partition"]["type"] == "partition"
        assert d["within_partition"]["attr"] == "region"

    def test_parse_partition_with_pred(self):
        """Partition with a predicate: ACROSS dept > 10."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > 50000 ACROSS dept > 10")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        part = d["across_partition"]
        assert part["type"] == "partition"
        assert part["attr"] == "dept"
        assert part["pred"]["type"] == "partition_pred"
        assert part["pred"]["comparator"] == ">"
        assert part["pred"]["value"] == 10.0

    def test_parse_cross_partition(self):
        """Cross partition: ACROSS dept X region."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > 50000 ACROSS dept X region")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        part = d["across_partition"]
        assert part["type"] == "cross_partition"
        assert part["lhs"]["type"] == "partition"
        assert part["lhs"]["attr"] == "dept"
        assert part["rhs"]["type"] == "partition"
        assert part["rhs"]["attr"] == "region"

    def test_parse_rv_with_expr(self):
        """RV wrapping a plain expression (not estimand)."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("bootstrap(salary) BETWEEN (0, 100)")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["quantity"]["type"] == "rv"
        assert d["event"]["quantity"]["distribution"] == "bootstrap"
        assert d["event"]["quantity"]["estimand"]["type"] == "attr"
        assert d["event"]["quantity"]["estimand"]["name"] == "salary"

    def test_parse_referent_func(self):
        """Function call as referent: salary > AVG(baseline)."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("salary > AVG(baseline)")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["referent"]["type"] == "func"
        assert d["event"]["referent"]["name"] == "AVG"
        assert len(d["event"]["referent"]["args"]) == 1

    def test_parse_binary_expr_chain(self):
        """Binary expression chain: AVG(x) + AVG(y) > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("AVG(x) + AVG(y) > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        qty = d["event"]["quantity"]
        assert qty["type"] == "binary"
        assert qty["lhs"]["type"] == "func"
        assert qty["lhs"]["name"] == "AVG"
        assert qty["op"] == "+"
        assert qty["rhs"]["type"] == "func"
        assert qty["rhs"]["name"] == "AVG"

    def test_parse_quantile_func(self):
        """QUANTILE function is recognized."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("QUANTILE(salary) > 0.5")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["quantity"]["type"] == "func"
        assert d["event"]["quantity"]["name"] == "QUANTILE"

    def test_parse_nested_conditioned_expr_in_expectation(self):
        """Nested conditioned expressions in expectation:
        E[(salary (dept = "eng") - salary (dept = "sci")) (year > 2020)] > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis(
            'E[(salary (dept = "eng") - salary (dept = "sci")) (year > 2020)] > 0'
        )
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["quantity"]["type"] == "expectation"
        # The expr inside E[...] is a conditioned_expr with outer predicate year > 2020
        inner = d["event"]["quantity"]["expr"]
        assert inner["type"] == "conditioned_expr"
        assert inner["predicate"]["attr"] == "year"
        assert inner["predicate"]["comparator"] == ">"
        # The subject is a binary expr of two conditioned sub-expressions
        subject = inner["expr"]
        assert subject["type"] == "binary"
        assert subject["op"] == "-"

    def test_parse_paren_grouped_subject(self):
        """Optional grouping parens around complex subject:
        (AVG(x) + MAX(y)) (z > 5) > 0."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("(AVG(x) + MAX(y)) (z > 5) > 0")
        assert parse_result.errors == []
        d = hypothesis_to_dict(parse_result.hypothesis)

        qty = d["event"]["quantity"]
        assert qty["type"] == "conditioned_expr"
        assert qty["predicate"]["attr"] == "z"
        # The subject is a binary expr inside grouping parens
        assert qty["expr"]["type"] == "binary"
        assert qty["expr"]["lhs"]["type"] == "func"
        assert qty["expr"]["lhs"]["name"] == "AVG"
        assert qty["expr"]["rhs"]["type"] == "func"
        assert qty["expr"]["rhs"]["name"] == "MAX"


class TestSICheckers:
    """Tests for the canonical field semantic invariant checkers."""

    # -- IR Fixtures --

    SIMPLE_EXPECTATION_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "expectation",
                "expr": {
                    "type": "conditioned_expr",
                    "expr": {"type": "attr", "name": "x"},
                    "predicate": {"type": "predicate", "kind": "comparison", "attr": "y", "comparator": "=", "value": 1},
                },
            },
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
        },
    }

    CONTRAST_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "contrast",
                "lhs": {
                    "type": "expectation",
                    "expr": {
                        "type": "conditioned_expr",
                        "expr": {"type": "attr", "name": "x"},
                        "predicate": {"type": "predicate", "kind": "comparison", "attr": "g", "comparator": "=", "value": 1},
                    },
                },
                "op": "-",
                "rhs": {
                    "type": "expectation",
                    "expr": {
                        "type": "conditioned_expr",
                        "expr": {"type": "attr", "name": "x"},
                        "predicate": {"type": "predicate", "kind": "comparison", "attr": "g", "comparator": "=", "value": 0},
                    },
                },
            },
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
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
                    "expr": {
                        "type": "conditioned_expr",
                        "expr": {"type": "attr", "name": "x"},
                        "predicate": {"type": "predicate", "kind": "comparison", "attr": "y", "comparator": "=", "value": 1},
                    },
                },
            },
            "comparator": "BETWEEN",
            "referent": {"type": "const", "value": [0, 1]},
        },
    }

    UNDERSPECIFIED_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
            "comparator": "\u16A6",
            "referent": {"type": "const", "value": 5},
        },
    }

    EMPTY_EVENT_IR = {"type": "hypothesis", "event": {}}

    CONDITIONED_EVENT_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "expectation",
                "expr": {
                    "type": "conditioned_expr",
                    "expr": {"type": "attr", "name": "x"},
                    "predicate": {
                        "type": "predicate",
                        "kind": "conjunction",
                        "lhs": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 1},
                        "rhs": {"type": "predicate", "kind": "comparison", "attr": "b", "comparator": "=", "value": 2},
                    },
                },
            },
            "comparator": ">",
            "referent": {"type": "const", "value": 5},
        },
    }

    QUANTITY_REFERENT_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {
                "type": "expectation",
                "expr": {
                    "type": "conditioned_expr",
                    "expr": {"type": "attr", "name": "x"},
                    "predicate": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 1},
                },
            },
            "comparator": ">",
            "referent": {
                "type": "expectation",
                "expr": {
                    "type": "conditioned_expr",
                    "expr": {"type": "attr", "name": "x"},
                    "predicate": {"type": "predicate", "kind": "comparison", "attr": "a", "comparator": "=", "value": 0},
                },
            },
        },
    }

    # -- ComparatorChecker tests --

    def test_comparator_extracts_valid(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == {"direction": "greater", "strictness": "strict", "polarity": "na", "form": "binary"}
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
        # Invalid operator gets unspecified decomposition
        assert result.value == {"direction": "none", "strictness": "na", "polarity": "na", "form": "unspecified"}
        assert result.exists is False

    def test_comparator_thorn_is_unspecified(self):
        """Thorn character should be extracted as unspecified decomposition."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        result = checker.extract_from_ir(self.UNDERSPECIFIED_IR)
        assert result.value == {"direction": "none", "strictness": "na", "polarity": "na", "form": "unspecified"}
        assert result.exists is True

    def test_comparator_existence_violation_on_missing(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        violations = checker.check(ir=self.EMPTY_EVENT_IR)
        ids = [v.invariantID for v in violations]
        assert "comparator-EX-IR" in ids

    def test_comparator_no_violation_on_valid(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker

        checker = ComparatorChecker()
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR)
        ex_violations = [v for v in violations if "EX-IR" in v.invariantID and "comparator" in v.invariantID]
        assert len(ex_violations) == 0

    # -- ReferentChecker tests --

    def test_referent_extracts_constant(self):
        """Const referent should be categorized as 'constant'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == {"type": "constant", "value": 0}
        assert result.exists is True

    def test_referent_extracts_absent_for_unspecified(self):
        """Unspecified referent should be categorized as 'absent'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "unspecified"},
            },
        }
        checker = ReferentChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == {"type": "absent", "value": "\u16A6"}
        assert result.exists is True
        assert result.metadata.get("unspecified") is True

    def test_referent_extracts_absent_for_none(self):
        """None referent should be categorized as 'absent'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": None,
            },
        }
        checker = ReferentChecker()
        result = checker.extract_from_ir(ir)
        assert result.value == {"type": "absent", "value": "\u16A6"}
        assert result.exists is True

    def test_referent_extracts_computed(self):
        """Quantity referent should be categorized as 'computed'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_ir(self.QUANTITY_REFERENT_IR)
        assert result.value["type"] == "computed"
        assert result.exists is True

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

    # -- ConditionsChecker tests --

    def test_conditions_extracts_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.exists is True
        assert len(result.value) == 1
        assert result.value[0]["attr"] == "y"
        assert result.value[0]["role"] == "filter"

    def test_conditions_contrast_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_ir(self.CONTRAST_IR)
        assert result.exists is True
        assert len(result.value) == 2
        # g appears on both arms → filter (canonical binding rule)
        roles = {c["attr"]: c["role"] for c in result.value}
        assert roles["g"] == "filter"

    def test_conditions_no_predicates(self):
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
            },
        }
        checker = ConditionsChecker()
        result = checker.extract_from_ir(ir)
        assert result.exists is False
        assert result.value == []

    def test_conditions_no_existence_violation(self):
        """Conditions are optional -- absence should not generate a violation."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
            },
        }
        checker = ConditionsChecker()
        violations = checker.check(ir=ir)
        ex_violations = [v for v in violations if "EX-" in v.invariantID]
        assert len(ex_violations) == 0

    def test_conditions_includes_compound_predicates(self):
        """Compound predicate should produce all constituent conditions."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_ir(self.CONDITIONED_EVENT_IR)
        attrs = [c["attr"] for c in result.value]
        assert "a" in attrs
        assert "b" in attrs

    def test_conditions_across_partition(self):
        """ACROSS partition attrs should get partition-key role."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
            },
            "across_partition": {"type": "partition", "attr": "dept", "pred": None},
        }
        checker = ConditionsChecker()
        result = checker.extract_from_ir(ir)
        assert result.exists is True
        roles = {c["attr"]: c["role"] for c in result.value}
        assert roles["dept"] == "partition-key"

    def test_conditions_within_partition(self):
        """WITHIN partition attrs should get stratification-key role."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        ir = {
            "type": "hypothesis",
            "event": {
                "type": "comparison",
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "const", "value": 0},
            },
            "within_partition": {"type": "partition", "attr": "region", "pred": None},
        }
        checker = ConditionsChecker()
        result = checker.extract_from_ir(ir)
        assert result.exists is True
        roles = {c["attr"]: c["role"] for c in result.value}
        assert roles["region"] == "stratification-key"

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
                    "lhs": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                    "op": "/",
                    "rhs": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
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
                        "lhs": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                        "op": "-",
                        "rhs": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
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
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, decompose_comparator

        checker = ComparatorChecker()
        nl_values = {"comparator": ExtractedValue(value=decompose_comparator(">"), exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_comparator_mismatch(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, decompose_comparator

        checker = ComparatorChecker()
        nl_values = {"comparator": ExtractedValue(value=decompose_comparator("<"), exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        # Direction sub-field mismatch: less vs greater
        assert len(pw_violations) == 1
        assert pw_violations[0].invariantID == "comparator.direction-PW-NLIR"
        assert pw_violations[0].expected == "less"
        assert pw_violations[0].observed == "greater"

    def test_pairwise_skipped_when_source_missing(self):
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue

        checker = ComparatorChecker()
        nl_values = {"comparator": ExtractedValue(value=None, exists=False)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-" in v.invariantID]
        assert len(pw_violations) == 0

    # -- Pairwise with nl_values dict --

    def test_pairwise_referent_match(self):
        """NL 'constant' should match IR 'constant'."""
        from campsite.campsite_lib.si_checkers import ReferentChecker, ExtractedValue

        checker = ReferentChecker()
        nl_values = {"referent": ExtractedValue(value={"type": "constant", "value": 0}, exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_referent_mismatch(self):
        """NL 'computed' should NOT match IR 'constant' type."""
        from campsite.campsite_lib.si_checkers import ReferentChecker, ExtractedValue

        checker = ReferentChecker()
        nl_values = {"referent": ExtractedValue(value={"type": "computed", "value": "\u16A6"}, exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        # At least type sub-field mismatch
        assert len(pw_violations) >= 1

    def test_pairwise_shape_match(self):
        """NL 'difference' should match IR 'difference'."""
        from campsite.campsite_lib.si_checkers import ShapeChecker, ExtractedValue

        checker = ShapeChecker()
        nl_values = {"quantity.shape": ExtractedValue(value="difference", exists=True)}
        violations = checker.check(ir=self.CONTRAST_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_uncertainty_match(self):
        """NL 'attached' should match IR 'attached'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker, ExtractedValue

        checker = UncertaintyChecker()
        nl_values = {"quantity.uncertainty": ExtractedValue(value="attached", exists=True)}
        violations = checker.check(ir=self.RV_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_uncertainty_mismatch(self):
        """NL 'attached' should NOT match IR 'missing'."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker, ExtractedValue

        checker = UncertaintyChecker()
        nl_values = {"quantity.uncertainty": ExtractedValue(value="attached", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 1

    # ConditionsChecker -- pairwise with role-aware records

    def test_pairwise_conditions_both_dicts_match(self):
        """Two record lists with same attrs and matching roles should match."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker, ExtractedValue

        checker = ConditionsChecker()
        nl_values = {"conditions": ExtractedValue(
            value=[
                {"attr": "b", "role": "filter", "values": "2", "ordered": False},
                {"attr": "a", "role": "filter", "values": "1", "ordered": False},
            ],
            exists=True,
        )}
        violations = checker.check(ir=self.CONDITIONED_EVENT_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) == 0

    def test_pairwise_conditions_attr_mismatch(self):
        """Different attrs should produce MISSING violations."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker, ExtractedValue

        checker = ConditionsChecker()
        nl_values = {"conditions": ExtractedValue(
            value=[{"attr": "z", "role": "filter", "values": "9", "ordered": False}],
            exists=True,
        )}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID or "MISSING" in v.invariantID]
        assert len(pw_violations) >= 1

    def test_pairwise_conditions_role_mismatch(self):
        """Same attr with different roles should produce role violation."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker, ExtractedValue

        checker = ConditionsChecker()
        nl_values = {"conditions": ExtractedValue(
            value=[{"attr": "y", "role": "partition-key", "values": "1", "ordered": False}],
            exists=True,
        )}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        role_violations = [v for v in violations if "conditions[y].role" in v.invariantID]
        assert len(role_violations) == 1

    # -- Confidence threshold tests --

    def test_low_confidence_downgrades_to_warn(self):
        """Low-confidence NL extraction should produce WARN, not FAIL, on mismatch."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, Criticality, decompose_comparator

        checker = ComparatorChecker()
        nl_values = {"comparator": ExtractedValue(value=decompose_comparator("<"), exists=True, confidence=0.2)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) >= 1
        assert pw_violations[0].criticality == Criticality.WARN

    def test_high_confidence_stays_fail(self):
        """High-confidence NL extraction should keep FAIL on mismatch."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, Criticality, decompose_comparator

        checker = ComparatorChecker()
        nl_values = {"comparator": ExtractedValue(value=decompose_comparator("<"), exists=True, confidence=0.9)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw_violations) >= 1
        assert pw_violations[0].criticality == Criticality.FAIL

    # -- ScopeChecker tests --

    ACROSS_PARTITION_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
        },
        "across_partition": {"type": "partition", "attr": "dept"},
    }

    WITHIN_PARTITION_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
        },
        "within_partition": {"type": "partition", "attr": "region"},
    }

    CROSS_PARTITION_IR = {
        "type": "hypothesis",
        "event": {
            "type": "comparison",
            "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
            "comparator": ">",
            "referent": {"type": "const", "value": 0},
        },
        "across_partition": {
            "type": "cross_partition",
            "lhs": {"type": "partition", "attr": "dept"},
            "rhs": {"type": "partition", "attr": "region"},
        },
    }

    def test_scope_extracts_none_without_partition(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker

        checker = ScopeChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "none"
        assert result.exists is True

    def test_scope_extracts_grouped_with_across(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker

        checker = ScopeChecker()
        result = checker.extract_from_ir(self.ACROSS_PARTITION_IR)
        assert result.value == "grouped"
        assert result.exists is True

    def test_scope_pairwise_violation_on_mismatch(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker, ExtractedValue

        checker = ScopeChecker()
        nl_values = {"scope": ExtractedValue(value="grouped", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw) == 1

    def test_scope_no_violation_when_matching(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker, ExtractedValue

        checker = ScopeChecker()
        nl_values = {"scope": ExtractedValue(value="grouped", exists=True)}
        violations = checker.check(ir=self.ACROSS_PARTITION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw) == 0

    # -- FrameChecker tests --

    def test_frame_extracts_none_without_within(self):
        from campsite.campsite_lib.si_checkers import FrameChecker

        checker = FrameChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.value == "none"
        assert result.exists is True

    def test_frame_extracts_stratified_with_within(self):
        from campsite.campsite_lib.si_checkers import FrameChecker

        checker = FrameChecker()
        result = checker.extract_from_ir(self.WITHIN_PARTITION_IR)
        assert result.value == "stratified"
        assert result.exists is True

    def test_frame_pairwise_violation_on_mismatch(self):
        from campsite.campsite_lib.si_checkers import FrameChecker, ExtractedValue

        checker = FrameChecker()
        nl_values = {"frame": ExtractedValue(value="stratified", exists=True)}
        violations = checker.check(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw) == 1

    def test_frame_no_violation_when_matching(self):
        from campsite.campsite_lib.si_checkers import FrameChecker, ExtractedValue

        checker = FrameChecker()
        nl_values = {"frame": ExtractedValue(value="stratified", exists=True)}
        violations = checker.check(ir=self.WITHIN_PARTITION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw) == 0

    # -- PartitionChecker tests --

    def test_partition_extracts_none_without_across(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_ir(self.SIMPLE_EXPECTATION_IR)
        assert result.exists is False

    def test_partition_extracts_single(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_ir(self.ACROSS_PARTITION_IR)
        assert result.value == {"structure": "single", "ordered": False}
        assert result.exists is True

    def test_partition_extracts_crossed(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_ir(self.CROSS_PARTITION_IR)
        assert result.value["structure"] == "crossed"
        assert result.exists is True

    def test_partition_pairwise_sub_field_violations(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker, ExtractedValue

        checker = PartitionChecker()
        nl_values = {
            "partition": ExtractedValue(
                value={"structure": "crossed", "ordered": True},
                exists=True,
            ),
        }
        # IR has single/False — should get violations on both sub-fields
        violations = checker.check(ir=self.ACROSS_PARTITION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        ids = {v.invariantID for v in pw}
        assert "partition.structure-PW-NLIR" in ids
        assert "partition.ordered-PW-NLIR" in ids

    def test_partition_no_violation_when_matching(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker, ExtractedValue

        checker = PartitionChecker()
        nl_values = {
            "partition": ExtractedValue(
                value={"structure": "single", "ordered": False},
                exists=True,
            ),
        }
        violations = checker.check(ir=self.ACROSS_PARTITION_IR, nl_values=nl_values)
        pw = [v for v in violations if "PW-NLIR" in v.invariantID]
        assert len(pw) == 0

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
        assert "comparator-EX-IR" in ids

    def test_runner_run_field(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_field("comparator", ir=self.SIMPLE_EXPECTATION_IR)
        # Should only return violations for that field
        for v in violations:
            assert "comparator" in v.invariantID

    def test_runner_get_checker(self):
        from campsite.campsite_lib.si_checkers import SICheckRunner, ComparatorChecker

        runner = SICheckRunner()
        checker = runner.get_checker("comparator")
        assert isinstance(checker, ComparatorChecker)

    def test_runner_with_nl_values(self):
        """Runner should accept nl_values dict for pairwise checking."""
        from campsite.campsite_lib.si_checkers import SICheckRunner, ExtractedValue, decompose_comparator

        runner = SICheckRunner()
        nl_values = {
            "comparator": ExtractedValue(value=decompose_comparator("<"), exists=True),
        }
        violations = runner.run_all(ir=self.SIMPLE_EXPECTATION_IR, nl_values=nl_values)
        pw_violations = [v for v in violations if "PW-NLIR" in v.invariantID and "comparator" in v.invariantID]
        assert len(pw_violations) >= 1

    # -- Violation serialization --

    def test_violation_to_dict(self):
        from campsite.campsite_lib.si_checkers import Violation, ViolationType, Criticality

        v = Violation(
            invariantID="comparator-EX-IR",
            violationType=ViolationType.MISSING_IN_IR,
            message="test",
            criticality=Criticality.WARN,
            expected="present",
            observed=None,
        )
        d = v.to_dict()
        assert d["invariantID"] == "comparator-EX-IR"
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

        parse_result = parse_hypothesis("E[x (y = 1)] > 0")
        assert parse_result.errors == []
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)
        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=ir_dict)
        ex_ir_violations = [v for v in violations if "EX-IR" in v.invariantID]
        assert len(ex_ir_violations) == 0

    def test_parsed_contrast_through_runner(self):
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict
        from campsite.campsite_lib.si_checkers import SICheckRunner

        parse_result = parse_hypothesis("E[x (g = 1)] - E[x (g = 0)] > 0")
        assert parse_result.errors == []
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)
        runner = SICheckRunner()
        violations = runner.run_ir_only(ir=ir_dict)
        ex_ir_violations = [v for v in violations if "EX-IR" in v.invariantID]
        assert len(ex_ir_violations) == 0

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
                "quantity": {"type": "expectation", "expr": {"type": "attr", "name": "x"}},
                "comparator": ">",
                "referent": {"type": "error", "boundary": "referent", "message": "bad ref"},
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

        result = _find_event_comparator("E[x (y = 1)] > 5")
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

        result = _find_event_comparator("bootstrap(E[x (y = 1)]) BETWEEN (0, 1)")
        assert result is not None
        assert result[0] == "BETWEEN"

    def test_thorn(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("E[x (a = 1)] \u16A6 5")
        assert result is not None
        assert result[0] == "\u16A6"

    def test_contrast_inner_ops_skipped(self):
        from campsite.campsite_lib.ir_parser import _find_event_comparator

        result = _find_event_comparator("E[x (g = 1)] - E[x (g = 0)] > 0")
        assert result is not None
        assert result[0] == ">"


class TestPartialRecovery:
    """Tests for partial parse recovery."""

    def test_recovery_bad_quantity_good_referent(self):
        """Malformed quantity but valid comparator and referent."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        # Old pipe syntax is invalid in new grammar; "> 5" is valid
        parse_result = parse_hypothesis("E[x | y = 1] > 5")
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

        parse_result = parse_hypothesis("E[x (y = 1)] > @@@bad")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "comparison"
        assert d["event"]["quantity"]["type"] == "expectation"
        qty_expr = d["event"]["quantity"]["expr"]
        assert qty_expr["type"] == "conditioned_expr"
        assert qty_expr["expr"]["name"] == "x"
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
        """No comparator at depth 0 -- falls back to full event ErrorNode."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("just random words")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert parse_result.is_partial
        assert d["event"]["type"] == "error"
        assert d["event"]["boundary"] == "event"

    def test_recovery_missing_referent(self):
        """Comparator found but nothing after -- quantity OK, referent error."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict

        parse_result = parse_hypothesis("E[x (y = 1)] >")
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

        parse_result = parse_hypothesis("E[x (y = 1)] >= @@@bad")
        d = hypothesis_to_dict(parse_result.hypothesis)

        assert d["event"]["type"] == "comparison"
        assert d["event"]["comparator"] == ">="
        assert d["event"]["quantity"]["type"] == "expectation"
        assert d["event"]["referent"]["type"] == "error"

    def test_successful_parse_returns_empty_errors(self):
        """Successful full parse returns ParseResult with empty errors."""
        from campsite.campsite_lib.ir_parser import parse_hypothesis

        parse_result = parse_hypothesis("E[x (y = 1)] > 0")
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
        from campsite.campsite_lib.ir_ast import (
            Hypothesis, Comparison, Expectation, Predicate, Const,
            ConditionedExpr, AttrVar,
        )

        cond_expr = ConditionedExpr(
            type="conditioned_expr",
            expr=AttrVar(name="x"),
            predicate=Predicate(type="predicate", kind="comparison", attr="y", comparator="=", value=1),
        )
        exp = Expectation(type="expectation", expr=cond_expr)
        ref = Const(type="const", value=0)
        comp = Comparison(type="comparison", quantity=exp, comparator=">", referent=ref)
        hyp = Hypothesis(type="hypothesis", event=comp)

        assert hyp.type == "hypothesis"
        assert hyp.event.type == "comparison"
        assert hyp.event.quantity.expr.expr.name == "x"
        assert hyp.event.referent.value == 0

    def test_contrast_dataclass(self):
        from campsite.campsite_lib.ir_ast import (
            Comparison, Expectation, Contrast, Const, AttrVar,
            ConditionedExpr, Predicate,
        )

        pred1 = Predicate(type="predicate", kind="comparison", attr="g", comparator="=", value=1)
        pred2 = Predicate(type="predicate", kind="comparison", attr="g", comparator="=", value=0)
        lhs = Expectation(type="expectation", expr=ConditionedExpr(expr=AttrVar(name="x"), predicate=pred1))
        rhs = Expectation(type="expectation", expr=ConditionedExpr(expr=AttrVar(name="x"), predicate=pred2))
        contrast = Contrast(type="contrast", lhs=lhs, op="-", rhs=rhs)
        comp = Comparison(type="comparison", quantity=contrast, comparator=">", referent=Const(value=0))

        assert comp.quantity.type == "contrast"
        assert comp.quantity.lhs.expr.expr.name == "x"
        assert comp.quantity.rhs.expr.expr.name == "x"

    def test_unspecified_dataclass(self):
        from campsite.campsite_lib.ir_ast import Comparison, Expectation, Unspecified, AttrVar

        exp = Expectation(type="expectation", expr=AttrVar(name="x"))
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

    def test_partition_dataclass(self):
        from campsite.campsite_lib.ir_ast import Partition, PartitionPred, CrossPartition

        part = Partition(type="partition", attr="dept", pred=None)
        assert part.type == "partition"
        assert part.attr == "dept"

        pred = PartitionPred(type="partition_pred", comparator=">", value=10)
        part_with_pred = Partition(type="partition", attr="dept", pred=pred)
        assert part_with_pred.pred.comparator == ">"

        cross = CrossPartition(
            type="cross_partition",
            lhs=Partition(attr="dept"),
            rhs=Partition(attr="region"),
        )
        assert cross.type == "cross_partition"
        assert cross.lhs.attr == "dept"
        assert cross.rhs.attr == "region"

    def test_hypothesis_with_partitions(self):
        from campsite.campsite_lib.ir_ast import Hypothesis, Comparison, Const, AttrVar, Partition

        comp = Comparison(
            type="comparison",
            quantity=AttrVar(name="salary"),
            comparator=">",
            referent=Const(value=50000),
        )
        hyp = Hypothesis(
            type="hypothesis",
            event=comp,
            across_partition=Partition(attr="dept"),
            within_partition=Partition(attr="region"),
        )
        assert hyp.across_partition.attr == "dept"
        assert hyp.within_partition.attr == "region"


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
        # "slope" now maps to "level" (trend removed)
        assert extractor.normalize("slope") == "level"

    def test_shape_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import ShapeNLExtractor

        extractor = ShapeNLExtractor()
        assert extractor.normalize("subtraction") == "difference"
        assert extractor.normalize("division") == "ratio"
        assert extractor.normalize("simple") == "value"
        assert extractor.normalize("ranking") == "rank"

    def test_uncertainty_normalize_aliases(self):
        from campsite.campsite_lib.nl_extractors import UncertaintyNLExtractor

        extractor = UncertaintyNLExtractor()
        assert extractor.normalize("confidence interval") == "attached"
        assert extractor.normalize("none") == "missing"
        assert extractor.normalize("point") == "missing"
        # "detached" now maps to "missing" (removed as standalone value)
        assert extractor.normalize("detached") == "missing"

    def test_nl_extractor_orchestrator(self):
        """NLExtractor.extract_all should return a dict with all field IDs."""
        from campsite.campsite_lib.nl_extractors import NLExtractor

        extractor = NLExtractor()
        # Verify new field IDs
        assert extractor.get_extractor("comparator") is not None
        assert extractor.get_extractor("referent") is not None
        assert extractor.get_extractor("quantity.signature") is not None
        assert extractor.get_extractor("conditions") is not None
        assert extractor.get_extractor("quantity.shape") is not None
        assert extractor.get_extractor("quantity.uncertainty") is not None
        assert extractor.get_extractor("scope") is not None
        assert extractor.get_extractor("partition") is not None
        assert extractor.get_extractor("frame") is not None
        # Old field IDs should not exist
        assert extractor.get_extractor("event.comparator") is None
        assert extractor.get_extractor("event.form") is None
        assert extractor.get_extractor("nonexistent") is None


class TestVegaUtils:
    """Tests for vega-lite spec traversal utilities."""

    def test_walk_units_flat_spec(self):
        """Single unit spec returns itself."""
        from campsite.campsite_lib.vega_utils import walk_units

        spec = {"mark": "point", "encoding": {"x": {"field": "a"}}}
        units = walk_units(spec)
        assert len(units) == 1
        assert units[0] is spec

    def test_walk_units_layered(self):
        """Layered spec returns all layer children."""
        from campsite.campsite_lib.vega_utils import walk_units

        spec = {
            "layer": [
                {"mark": "point", "encoding": {}},
                {"mark": "rule", "encoding": {}},
            ]
        }
        units = walk_units(spec)
        assert len(units) == 2

    def test_walk_units_hconcat(self):
        """hconcat spec returns all children."""
        from campsite.campsite_lib.vega_utils import walk_units

        spec = {
            "hconcat": [
                {"mark": "point", "encoding": {}},
                {"mark": "point", "encoding": {}},
            ]
        }
        units = walk_units(spec)
        assert len(units) == 2

    def test_walk_units_nested(self):
        """Nested composition (layer inside hconcat) returns all leaf units."""
        from campsite.campsite_lib.vega_utils import walk_units

        spec = {
            "hconcat": [
                {"layer": [{"mark": "point", "encoding": {}}, {"mark": "rule", "encoding": {}}]},
                {"mark": "line", "encoding": {}},
            ]
        }
        units = walk_units(spec)
        assert len(units) == 3

    def test_walk_units_empty(self):
        """Empty dict returns empty list."""
        from campsite.campsite_lib.vega_utils import walk_units

        assert walk_units({}) == []

    def test_get_mark_type_string(self):
        """String mark shorthand."""
        from campsite.campsite_lib.vega_utils import get_mark_type

        assert get_mark_type({"mark": "point"}) == "point"

    def test_get_mark_type_object(self):
        """Object mark form."""
        from campsite.campsite_lib.vega_utils import get_mark_type

        assert get_mark_type({"mark": {"type": "area", "opacity": 0.5}}) == "area"

    def test_get_mark_type_missing(self):
        """No mark key returns None."""
        from campsite.campsite_lib.vega_utils import get_mark_type

        assert get_mark_type({"encoding": {}}) is None

    def test_get_transforms_walks_tree(self):
        """Transforms at different composition levels are all collected."""
        from campsite.campsite_lib.vega_utils import get_transforms

        spec = {
            "transform": [{"filter": "datum.x > 0"}],
            "layer": [
                {"mark": "point", "encoding": {}, "transform": [{"calculate": "datum.a + 1", "as": "b"}]},
                {"mark": "line", "encoding": {}},
            ],
        }
        transforms = get_transforms(spec)
        assert len(transforms) == 2
        assert transforms[0].get("filter") == "datum.x > 0"
        assert transforms[1].get("calculate") == "datum.a + 1"

    def test_detect_referent_excludes_uncertainty_bands(self):
        """Area with y+y2 (confidence band) should not be counted as a separate quantity."""
        from campsite.campsite_lib.vega_utils import detect_referent_pattern

        spec = {
            "layer": [
                {"mark": "line", "encoding": {"x": {"field": "date"}, "y": {"field": "mean"}}},
                {"mark": "area", "encoding": {"x": {"field": "date"}, "y": {"field": "lo"}, "y2": {"field": "hi"}}},
            ]
        }
        assert detect_referent_pattern(spec) == "missing"

    def test_has_uncertainty_encoding(self):
        """Area with y+y2 is detected as uncertainty."""
        from campsite.campsite_lib.vega_utils import has_uncertainty_encoding

        spec = {
            "layer": [
                {"mark": "line", "encoding": {"y": {"field": "mean"}}},
                {"mark": "area", "encoding": {"y": {"field": "lo"}, "y2": {"field": "hi"}}},
            ]
        }
        assert has_uncertainty_encoding(spec) is True

    def test_has_uncertainty_encoding_none(self):
        """Simple spec has no uncertainty encoding."""
        from campsite.campsite_lib.vega_utils import has_uncertainty_encoding

        spec = {"mark": "point", "encoding": {"y": {"field": "val"}}}
        assert has_uncertainty_encoding(spec) is False

    def test_get_aggregate_groupby_fields(self):
        """Groupby fields are extracted from aggregate transforms."""
        from campsite.campsite_lib.vega_utils import get_aggregate_groupby_fields

        spec = {
            "transform": [
                {"aggregate": [{"op": "mean", "field": "val", "as": "mean_val"}], "groupby": ["cat", "grp"]}
            ],
            "mark": "point",
            "encoding": {},
        }
        fields = get_aggregate_groupby_fields(spec)
        assert fields == ["cat", "grp"]

    def test_has_division_calculate(self):
        """Calculate transform with / is detected."""
        from campsite.campsite_lib.vega_utils import has_division_calculate

        spec = {"transform": [{"calculate": "datum.a / datum.b", "as": "ratio"}], "mark": "point"}
        assert has_division_calculate(spec) is True

    def test_has_division_calculate_none(self):
        """Spec without calculate transform returns False."""
        from campsite.campsite_lib.vega_utils import has_division_calculate

        spec = {"mark": "point", "encoding": {}}
        assert has_division_calculate(spec) is False

    def test_get_facet_fields(self):
        """Row and column encoding fields are detected."""
        from campsite.campsite_lib.vega_utils import get_facet_fields

        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "x"},
                "row": {"field": "region", "type": "nominal"},
            },
        }
        assert get_facet_fields(spec) == ["region"]


class TestVisArtifactExtractors:
    """Tests for visualization artifact extraction from vega-lite specs."""

    # -- Fixture specs (inline, matching TestSICheckers pattern) --

    SIMPLE_POINT = {
        "mark": "point",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative", "aggregate": "mean"},
        },
    }

    LINE_TREND = {
        "mark": "line",
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "price", "type": "quantitative"},
        },
    }

    GROUPED_POINT = {
        "mark": "point",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative", "aggregate": "mean"},
            "color": {"field": "group", "type": "nominal"},
        },
    }

    LAYERED_WITH_RULE = {
        "layer": [
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "category", "type": "nominal"},
                    "y": {"field": "value", "type": "quantitative", "aggregate": "mean"},
                },
            },
            {"mark": "rule", "encoding": {"y": {"datum": 50}}},
        ],
    }

    SCATTERPLOT = {
        "mark": "point",
        "encoding": {
            "x": {"field": "height", "type": "quantitative"},
            "y": {"field": "weight", "type": "quantitative"},
        },
    }

    DENSITY_AREA = {
        "transform": [{"density": "value"}],
        "mark": "area",
        "encoding": {
            "x": {"field": "value", "type": "quantitative"},
            "y": {"field": "density", "type": "quantitative"},
        },
    }

    CONFIDENCE_BAND = {
        "layer": [
            {
                "mark": "line",
                "encoding": {
                    "x": {"field": "date", "type": "temporal"},
                    "y": {"field": "mean", "type": "quantitative"},
                },
            },
            {
                "mark": "area",
                "encoding": {
                    "x": {"field": "date", "type": "temporal"},
                    "y": {"field": "ci_lower", "type": "quantitative"},
                    "y2": {"field": "ci_upper"},
                },
            },
        ],
    }

    HCONCAT_COMPARISON = {
        "hconcat": [
            {
                "transform": [{"filter": "datum.group === 'A'"}],
                "mark": "point",
                "encoding": {
                    "x": {"field": "category", "type": "nominal"},
                    "y": {"field": "value", "type": "quantitative", "aggregate": "mean"},
                },
            },
            {
                "transform": [{"filter": "datum.group === 'B'"}],
                "mark": "point",
                "encoding": {
                    "x": {"field": "category", "type": "nominal"},
                    "y": {"field": "value", "type": "quantitative", "aggregate": "mean"},
                },
            },
        ],
    }

    RATIO_CALCULATE = {
        "transform": [{"calculate": "datum.a / datum.b", "as": "ratio"}],
        "mark": "point",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "ratio", "type": "quantitative"},
        },
    }

    FACETED_BY_ROW = {
        "mark": "point",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative", "aggregate": "mean"},
            "row": {"field": "region", "type": "nominal"},
        },
    }

    STRIP_PLOT = {
        "mark": "point",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    AREA_TEMPORAL = {
        "mark": "area",
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "count", "type": "quantitative"},
        },
    }

    RULE_ERROR_BARS = {
        "layer": [
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "cat", "type": "nominal"},
                    "y": {"field": "mean", "type": "quantitative"},
                },
            },
            {
                "mark": "rule",
                "encoding": {
                    "x": {"field": "cat", "type": "nominal"},
                    "y": {"field": "lo", "type": "quantitative"},
                    "y2": {"field": "hi"},
                },
            },
        ],
    }

    LAYERED_WITH_TEXT = {
        "layer": [
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "cat", "type": "nominal"},
                    "y": {"field": "val", "type": "quantitative", "aggregate": "mean"},
                },
            },
            {
                "mark": "text",
                "encoding": {
                    "text": {"value": "target = 50"},
                },
            },
        ],
    }

    # -- ComparatorChecker tests --

    def test_comparator_artifact_always_unspecified(self):
        """Comparator from any spec returns unspecified decomposition."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, decompose_comparator

        checker = ComparatorChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == decompose_comparator(None)
        assert result.exists is True

    def test_comparator_artifact_pairwise_skips_art(self):
        """Comparator pairwise check skips ART-related pairs."""
        from campsite.campsite_lib.si_checkers import ComparatorChecker, ExtractedValue, decompose_comparator

        checker = ComparatorChecker()
        ir_val = ExtractedValue(value=decompose_comparator(">"), exists=True)
        art_val = ExtractedValue(value=decompose_comparator(None), exists=True)

        # IR-ART pairwise should be suppressed
        assert checker.check_pairwise(ir_val, art_val, "PW-IRART") is None
        assert checker.check_pairwise(art_val, ir_val, "PW-NLART") is None

        # NL-IR pairwise should still work
        nl_val = ExtractedValue(value=decompose_comparator("<"), exists=True)
        violation = checker.check_pairwise(nl_val, ir_val, "PW-NLIR")
        assert violation is not None

    # -- ReferentChecker tests --

    def test_referent_artifact_rule_is_constant(self):
        """Rule mark indicates a constant referent."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_artifact(self.LAYERED_WITH_RULE)
        assert result.value["type"] == "constant"

    def test_referent_artifact_text_is_constant(self):
        """Text mark indicates a constant referent (annotation)."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_artifact(self.LAYERED_WITH_TEXT)
        assert result.value["type"] == "constant"

    def test_referent_artifact_hconcat_is_computed(self):
        """hconcat indicates computed referent comparison."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_artifact(self.HCONCAT_COMPARISON)
        assert result.value["type"] == "computed"

    def test_referent_artifact_multi_layer_is_computed(self):
        """Multiple primary mark layers indicate computed referent."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        spec = {
            "layer": [
                {"mark": "area", "encoding": {"x": {"field": "v", "type": "quantitative"}, "y": {"field": "d"}}},
                {"mark": "area", "encoding": {"x": {"field": "v", "type": "quantitative"}, "y": {"field": "d2"}}},
            ]
        }
        checker = ReferentChecker()
        result = checker.extract_from_artifact(spec)
        assert result.value["type"] == "computed"

    def test_referent_artifact_single_mark_is_absent(self):
        """Single mark spec has absent referent."""
        from campsite.campsite_lib.si_checkers import ReferentChecker

        checker = ReferentChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value["type"] == "absent"

    # -- QuantitySignatureChecker tests --

    def test_signature_artifact_line_is_level(self):
        """Line mark -> level (trend removed)."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.LINE_TREND)
        assert result.value == "level"

    def test_signature_artifact_area_temporal_is_level(self):
        """Area with temporal x -> level (trend removed)."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.AREA_TEMPORAL)
        assert result.value == "level"

    def test_signature_artifact_area_quantitative_is_distribution(self):
        """Area with quantitative x -> distribution."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.DENSITY_AREA)
        assert result.value == "distribution"

    def test_signature_artifact_scatterplot_is_association(self):
        """Point with both axes quantitative, no aggregation -> association."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.SCATTERPLOT)
        assert result.value == "association"

    def test_signature_artifact_point_strip_is_distribution(self):
        """Point with nominal x, quantitative y, no aggregation -> distribution."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.STRIP_PLOT)
        assert result.value == "distribution"

    def test_signature_artifact_grouped_is_contrast(self):
        """Grouped point (color encoding) -> contrast."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.GROUPED_POINT)
        assert result.value == "contrast"

    def test_signature_artifact_single_aggregated_is_level(self):
        """Single aggregated point with no grouping -> level."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == "level"

    def test_signature_artifact_empty_spec(self):
        """Empty spec returns not-found."""
        from campsite.campsite_lib.si_checkers import QuantitySignatureChecker

        checker = QuantitySignatureChecker()
        result = checker.extract_from_artifact({})
        assert result.exists is False

    # -- ConditionsChecker tests --

    def test_conditions_artifact_no_conditions(self):
        """Simple spec with no conditions returns empty list."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == []
        assert result.exists is False

    def test_conditions_artifact_filter_transforms(self):
        """Filter transforms are extracted as filter-role conditions."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_artifact(self.HCONCAT_COMPARISON)
        assert result.exists is True
        roles = {c["attr"]: c["role"] for c in result.value if isinstance(c, dict)}
        assert any(r == "filter" for r in roles.values())

    def test_conditions_artifact_facet_fields(self):
        """Row/column encoding fields are extracted as stratification-key conditions."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_artifact(self.FACETED_BY_ROW)
        assert result.exists is True
        roles = {c["attr"]: c["role"] for c in result.value if isinstance(c, dict)}
        assert roles.get("region") == "stratification-key"

    def test_conditions_artifact_color_grouping(self):
        """Nominal color encoding is extracted as partition-key condition."""
        from campsite.campsite_lib.si_checkers import ConditionsChecker

        checker = ConditionsChecker()
        result = checker.extract_from_artifact(self.GROUPED_POINT)
        assert result.exists is True
        roles = {c["attr"]: c["role"] for c in result.value if isinstance(c, dict)}
        assert roles.get("group") == "partition-key"

    # -- ShapeChecker tests --

    def test_shape_artifact_ratio(self):
        """Calculate transform with division -> ratio."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        checker = ShapeChecker()
        result = checker.extract_from_artifact(self.RATIO_CALCULATE)
        assert result.value == "ratio"

    def test_shape_artifact_difference(self):
        """hconcat spec -> difference."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        checker = ShapeChecker()
        result = checker.extract_from_artifact(self.HCONCAT_COMPARISON)
        assert result.value == "difference"

    def test_shape_artifact_value(self):
        """Simple single-mark spec -> value."""
        from campsite.campsite_lib.si_checkers import ShapeChecker

        checker = ShapeChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == "value"

    # -- UncertaintyChecker tests --

    def test_uncertainty_artifact_attached_area_band(self):
        """Area with y+y2 (confidence band) -> attached."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker

        checker = UncertaintyChecker()
        result = checker.extract_from_artifact(self.CONFIDENCE_BAND)
        assert result.value == "attached"

    def test_uncertainty_artifact_attached_rule_errorbars(self):
        """Rule with y+y2 (error bars) -> attached."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker

        checker = UncertaintyChecker()
        result = checker.extract_from_artifact(self.RULE_ERROR_BARS)
        assert result.value == "attached"

    def test_uncertainty_artifact_missing(self):
        """Simple spec without y2 -> missing."""
        from campsite.campsite_lib.si_checkers import UncertaintyChecker

        checker = UncertaintyChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == "missing"

    # -- Integration tests --

    def test_runner_with_artifact(self):
        """SICheckRunner.run_all with IR + artifact runs without error."""
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        violations = runner.run_all(
            ir=TestSICheckers.SIMPLE_EXPECTATION_IR,
            artifact=self.SIMPLE_POINT,
        )
        # Should complete without exception and return a list
        assert isinstance(violations, list)

    def test_runner_with_artifact_pairwise_violations(self):
        """Mismatched IR and artifact produce IR-ART pairwise violations."""
        from campsite.campsite_lib.si_checkers import SICheckRunner

        runner = SICheckRunner()
        # IR has contrast (difference shape), artifact shows single point (value shape)
        violations = runner.run_all(
            ir=TestSICheckers.CONTRAST_IR,
            artifact=self.SIMPLE_POINT,
        )
        # Should have some pairwise IR-ART violations
        ir_art_violations = [v for v in violations if "IRART" in v.invariantID]
        assert len(ir_art_violations) > 0

    def test_runner_with_all_three_representations(self):
        """Run with IR + NL + artifact checks all pairwise combinations."""
        from campsite.campsite_lib.si_checkers import SICheckRunner, ExtractedValue, decompose_comparator

        runner = SICheckRunner()
        nl_values = {
            "comparator": ExtractedValue(value=decompose_comparator(">"), exists=True),
            "referent": ExtractedValue(value={"type": "constant", "value": 0}, exists=True),
            "quantity.signature": ExtractedValue(value="level", exists=True),
            "conditions": ExtractedValue(value=[], exists=False),
            "quantity.shape": ExtractedValue(value="value", exists=True),
            "quantity.uncertainty": ExtractedValue(value="missing", exists=True),
        }
        violations = runner.run_all(
            ir=TestSICheckers.SIMPLE_EXPECTATION_IR,
            nl_values=nl_values,
            artifact=self.LAYERED_WITH_RULE,
        )
        assert isinstance(violations, list)

    # -- ScopeChecker artifact tests --

    def test_scope_artifact_grouped_with_color(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker

        checker = ScopeChecker()
        result = checker.extract_from_artifact(self.GROUPED_POINT)
        assert result.value == "grouped"

    def test_scope_artifact_grouped_with_facet(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker

        checker = ScopeChecker()
        result = checker.extract_from_artifact(self.FACETED_BY_ROW)
        assert result.value == "grouped"

    def test_scope_artifact_none_without_grouping(self):
        from campsite.campsite_lib.si_checkers import ScopeChecker

        checker = ScopeChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == "none"

    # -- FrameChecker artifact tests --

    def test_frame_artifact_stratified_with_facet(self):
        from campsite.campsite_lib.si_checkers import FrameChecker

        checker = FrameChecker()
        result = checker.extract_from_artifact(self.FACETED_BY_ROW)
        assert result.value == "stratified"

    def test_frame_artifact_none_without_facet(self):
        from campsite.campsite_lib.si_checkers import FrameChecker

        checker = FrameChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.value == "none"

    # -- PartitionChecker artifact tests --

    GROUPED_BAR_WITH_AGGREGATE = {
        "transform": [
            {"aggregate": [{"op": "mean", "field": "val", "as": "mean_val"}], "groupby": ["dept"]},
        ],
        "mark": "bar",
        "encoding": {
            "x": {"field": "dept", "type": "nominal"},
            "y": {"field": "mean_val", "type": "quantitative"},
        },
    }

    CROSSED_GROUPING = {
        "transform": [
            {"aggregate": [{"op": "mean", "field": "val", "as": "mean_val"}], "groupby": ["dept", "region"]},
        ],
        "mark": "bar",
        "encoding": {
            "x": {"field": "dept", "type": "nominal"},
            "y": {"field": "mean_val", "type": "quantitative"},
            "color": {"field": "region", "type": "nominal"},
        },
    }

    ORDERED_GROUPING = {
        "mark": "line",
        "encoding": {
            "x": {"field": "month", "type": "temporal"},
            "y": {"field": "sales", "type": "quantitative"},
            "color": {"field": "product", "type": "nominal"},
        },
    }

    def test_partition_artifact_no_grouping(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_artifact(self.SIMPLE_POINT)
        assert result.exists is False

    def test_partition_artifact_single_color(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_artifact(self.GROUPED_POINT)
        assert result.exists is True
        assert result.value["structure"] == "single"

    def test_partition_artifact_single_aggregate(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_artifact(self.GROUPED_BAR_WITH_AGGREGATE)
        assert result.exists is True
        assert result.value["structure"] == "single"

    def test_partition_artifact_crossed(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_artifact(self.CROSSED_GROUPING)
        assert result.exists is True
        assert result.value["structure"] == "crossed"

    def test_partition_artifact_ordered_temporal(self):
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        result = checker.extract_from_artifact(self.ORDERED_GROUPING)
        assert result.exists is True
        assert result.value["ordered"] is True

    def test_partition_artifact_ir_pairwise_structure_mismatch(self):
        """IR says single, artifact says crossed → sub-field violation."""
        from campsite.campsite_lib.si_checkers import PartitionChecker

        checker = PartitionChecker()
        violations = checker.check(
            ir=TestSICheckers.ACROSS_PARTITION_IR,
            artifact=self.CROSSED_GROUPING,
        )
        ids = {v.invariantID for v in violations}
        assert "partition.structure-PW-IRART" in ids


class TestEncodingValidation:
    """Tests for encoding channel validation in artifact_gen."""

    def test_valid_channels_pass(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "a", "type": "nominal"},
                "y": {"field": "b", "type": "quantitative"},
                "color": {"field": "c", "type": "nominal"},
            },
        }
        assert validate_encodings(spec) == []

    def test_disallowed_channel_detected(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "a", "type": "nominal"},
                "size": {"field": "b", "type": "quantitative"},
            },
        }
        assert validate_encodings(spec) == ["size"]

    def test_multiple_disallowed_channels(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "a", "type": "nominal"},
                "size": {"field": "b", "type": "quantitative"},
                "opacity": {"value": 0.5},
                "shape": {"field": "c", "type": "nominal"},
            },
        }
        bad = validate_encodings(spec)
        assert set(bad) == {"size", "opacity", "shape"}

    def test_nested_layer_channels(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "layer": [
                {
                    "mark": "point",
                    "encoding": {"x": {"field": "a"}, "y": {"field": "b"}},
                },
                {
                    "mark": "line",
                    "encoding": {"x": {"field": "a"}, "tooltip": {"field": "b"}},
                },
            ]
        }
        assert validate_encodings(spec) == ["tooltip"]

    def test_no_encoding_key(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {"mark": "point"}
        assert validate_encodings(spec) == []

    def test_y2_and_text_allowed(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "layer": [
                {
                    "mark": "area",
                    "encoding": {
                        "x": {"field": "a"},
                        "y": {"field": "lo"},
                        "y2": {"field": "hi"},
                    },
                },
                {
                    "mark": "text",
                    "encoding": {
                        "x": {"field": "a"},
                        "y": {"field": "b"},
                        "text": {"field": "label"},
                    },
                },
            ]
        }
        assert validate_encodings(spec) == []

    def test_faceted_with_row_column(self):
        from campsite.campsite_lib.artifact_gen import validate_encodings

        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "a"},
                "y": {"field": "b"},
                "row": {"field": "r"},
                "column": {"field": "c"},
            },
        }
        assert validate_encodings(spec) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
