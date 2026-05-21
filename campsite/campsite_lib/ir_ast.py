"""AST dataclasses for the hypothesis intermediate representation."""

from dataclasses import dataclass, field
from typing import Union, Literal, Optional


# ---------- Constants ----------
@dataclass
class Const:
    """Constant value node."""

    type: Literal["const"] = "const"
    value: Union[int, float, str, tuple[float, float]] = 0


ConstValue = Union[int, float, str, tuple[float, float]]


# ---------- Wildcard (***) ----------
@dataclass
class Wildcard:
    """Wildcard predicate value (***) — matches any categorical level.

    Retained for backward compatibility; no longer produced by the parser.
    """

    type: Literal["wildcard"] = "wildcard"


# ---------- Unspecified (ᚦ) ----------
@dataclass
class Unspecified:
    """Represents an intentionally unspecified part of a hypothesis (ᚦ)."""

    type: Literal["unspecified"] = "unspecified"


# ---------- Error Handling ----------
@dataclass
class ErrorNode:
    """Error node for malformed input."""

    type: Literal["error"] = "error"
    boundary: Literal["quantity", "estimand", "predicate", "expr", "const", "event", "comparator", "referent"] = "event"
    message: str = ""
    text: str = ""
    start: int = 0
    end: int = 0


# ---------- Predicates ----------
@dataclass
class Predicate:
    """Predicate for conditional expressions."""

    type: Literal["predicate"] = "predicate"
    kind: Literal["comparison", "conjunction", "disjunction"] = "comparison"
    attr: str = ""
    comparator: str = "="
    value: "PredRef" = 0
    # For compound predicates (kind="conjunction" or "disjunction")
    lhs: Optional["Predicate"] = None
    rhs: Optional["Predicate"] = None


# ---------- Variables ----------
@dataclass
class AttrVar:
    """Attribute variable reference."""

    type: Literal["attr"] = "attr"
    name: str = ""


@dataclass
class ConstVar:
    """Constant variable."""

    type: Literal["const"] = "const"
    value: ConstValue = 0


Var = Union[AttrVar, ConstVar]


# ---------- Expressions ----------
@dataclass
class FuncCall:
    """Function call expression."""

    type: Literal["func"] = "func"
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class BinaryExpr:
    """Binary expression."""

    type: Literal["binary"] = "binary"
    lhs: "Expr" = None
    op: Literal["+", "-", "*", "/"] = "+"
    rhs: "Expr" = None


# ---------- Conditioned Expression ----------
@dataclass
class ConditionedExpr:
    """Expression conditioned on a predicate: subject (predicate)."""

    type: Literal["conditioned_expr"] = "conditioned_expr"
    expr: "Expr" = None
    predicate: Union[Predicate, ErrorNode] = None


Expr = Union[AttrVar, ConstVar, FuncCall, BinaryExpr, "Extract", ConditionedExpr, ErrorNode]


# ---------- Predicate value types ----------
PredRef = Union[ConstValue, "AttrVar", "ConstVar", "FuncCall", "BinaryExpr", "Extract", Wildcard]


# ---------- Estimands ----------
@dataclass
class Expectation:
    """Expectation estimand E[expr]."""

    type: Literal["expectation"] = "expectation"
    expr: Expr = None


@dataclass
class Contrast:
    """Contrast between two expectations."""

    type: Literal["contrast"] = "contrast"
    lhs: Union[Expectation, ErrorNode] = None
    op: Literal["+", "-", "*", "/"] = "-"
    rhs: Union[Expectation, ErrorNode] = None


Estimand = Union[Contrast, Expectation, ErrorNode]


# ---------- Uncertainty ----------
@dataclass
class RV:
    """Random variable wrapping an estimand or expression."""

    type: Literal["rv"] = "rv"
    distribution: str = ""
    estimand: Union[Estimand, Expr] = None


# ---------- Quantities ----------
Quantity = Union[RV, Estimand, Expr, ErrorNode]


# ---------- Partitions ----------
@dataclass
class PartitionPred:
    """Condition on a partition attribute restricting which groups participate."""

    type: Literal["partition_pred"] = "partition_pred"
    comparator: str = "="
    value: PredRef = None


@dataclass
class Partition:
    """Defines how data is split into groups by a single attribute."""

    type: Literal["partition"] = "partition"
    attr: str = ""
    pred: Optional[PartitionPred] = None


@dataclass
class CrossPartition:
    """Cartesian product of two partitions (X operator)."""

    type: Literal["cross_partition"] = "cross_partition"
    lhs: "PartitionType" = None
    rhs: "PartitionType" = None


PartitionType = Union[Partition, CrossPartition]


# ---------- Events ----------
@dataclass
class Comparison:
    """Event: quantity comparator referent."""

    type: Literal["comparison"] = "comparison"
    quantity: Quantity = None
    comparator: str = "="  # "ᚦ" when intentionally unspecified
    referent: Union[Const, FuncCall, Unspecified, ErrorNode] = None


Event = Union[Comparison, ErrorNode]


# ---------- Core ----------
@dataclass
class Hypothesis:
    """Top-level hypothesis node."""

    type: Literal["hypothesis"] = "hypothesis"
    event: Event = None
    across_partition: Optional[PartitionType] = None
    within_partition: Optional[PartitionType] = None


# ---------- Model extraction ----------
@dataclass
class Extract:
    """Model extraction node: Extract(model, expr)."""

    type: Literal["extract"] = "extract"
    model: str = ""
    expr: Expr = None


# ---------- Parse Result ----------
@dataclass
class ParseResult:
    """Result of parsing a hypothesis string.

    Wraps a Hypothesis AST with a list of ErrorNodes encountered during
    parsing. When errors is empty, the parse was fully successful.
    When errors is non-empty, partial recovery was attempted.
    """

    hypothesis: Hypothesis = None
    errors: list[ErrorNode] = field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        return len(self.errors) > 0


# Valid comparators (ᚦ = intentionally unspecified)
VALID_COMPARATORS = frozenset(["=", ">", "<", ">=", "<=", "!=", "BETWEEN", "IN", "ᚦ"])

# Function names
FUNC_NAMES = frozenset([
    "AVG", "MAX", "MIN", "SUM", "COUNT",
    "MEDIAN", "VARIANCE", "STDDEV",
    "CORR", "PERCENTILE", "QUANTILE"
])

# Field operators
FIELD_OPS = frozenset(["+", "-", "*", "/"])
